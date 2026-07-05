"""Per-country checkpoint evaluation: pixel, object, and polygon-level metrics
(PQ/SQ/RQ, COCO-grid object F1, boundary chamfer error) from a single D4 TTA +
watershed inference pass per patch. Models are scored as parcel-recovery
systems on vectorized predictions, not pixel maps.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio.features
import shapely.geometry
import torch
from ftw_tools.training.metrics import get_object_level_metrics
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet
from ftw_planet.inference import load_task, pad_to_min, predict, predict_tta, watershed_instances

S2_SCALE = 3000.0
GSD_M = {"planet": 3.0, "s2": 10.0}  # ground-sample distance in meters
AP_IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2).tolist()

CSV_COLUMNS = (
    "checkpoint",
    "country",
    "n_patches",
    "pixel_level_iou",
    "pixel_level_precision",
    "pixel_level_recall",
    "object_pix_precision",
    "object_pix_recall",
    "object_pix_f1",
    "object_ws_precision",
    "object_ws_recall",
    "object_ws_f1",
    "pq_sq",
    "pq_rq",
    "pq",
    "ap_5_95",
    "polygon_count_delta_mean",
    "polygon_count_delta_median",
    "boundary_error_m_mean",
    "boundary_error_m_p95",
)


def _make_dataset(country: str, root: str, split: str, backend: str) -> tuple[Any, float]:
    if backend == "s2":
        from ftw_tools.training.datasets import FTW

        ds = FTW(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
            temporal_options="stacked",
        )
        return ds, S2_SCALE
    ds = FTWPlanet(
        root=root, countries=[country], split=split, transforms=None, load_boundaries=True
    )
    return ds, PLANET_SR_SCALE


def _extract_shapes(mask: np.ndarray) -> list[shapely.geometry.base.BaseGeometry]:
    """Connected-component polygons of value==1 in ``mask`` (uint8)."""
    return [
        shapely.geometry.shape(geom)
        for geom, val in rasterio.features.shapes(mask.astype(np.uint8))
        if val == 1
    ]


def _match_shapes(
    gt_shapes: list, pred_shapes: list
) -> tuple[dict[float, tuple[int, int, int]], list[tuple[int, int, float]]]:
    """Greedy-match GT vs predicted polygons from a single pairwise-IoU matrix
    (get_object_level_metrics' greedy strategy: first prediction with
    IoU > threshold wins). Returns per-threshold (tp, fp, fn) counts and the
    (i, j, iou) pairs matched at the lowest threshold (for SQ and chamfer).
    """
    n_gt, n_pred = len(gt_shapes), len(pred_shapes)
    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, g in enumerate(gt_shapes):
        for j, p in enumerate(pred_shapes):
            if not g.intersects(p):
                continue
            inter = g.intersection(p).area
            union = g.area + p.area - inter
            if inter > 0 and union > 0:
                ious[i, j] = inter / union

    per_t: dict[float, tuple[int, int, int]] = {}
    matched_low: list[tuple[int, int, float]] = []
    for k, t in enumerate(AP_IOU_THRESHOLDS):
        matched_j: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for i in range(n_gt):
            for j in range(n_pred):
                if j not in matched_j and ious[i, j] > t:
                    matched_j.add(j)
                    pairs.append((i, j, float(ious[i, j])))
                    break
        per_t[t] = (len(pairs), n_pred - len(matched_j), n_gt - len(pairs))
        if k == 0:
            matched_low = pairs
    return per_t, matched_low


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """1-pixel-thick boundary of a binary mask (pixels within 1.5 px of edge)."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return mask.astype(bool) & (distance_transform_edt(mask) < 1.5)


def _symmetric_chamfer(pred_bd: np.ndarray, gt_bd: np.ndarray) -> float | None:
    """Mean symmetric chamfer distance (pixels) between two boundary masks."""
    if not pred_bd.any() or not gt_bd.any():
        return None
    d1 = float(distance_transform_edt(~pred_bd)[gt_bd].mean())
    d2 = float(distance_transform_edt(~gt_bd)[pred_bd].mean())
    return 0.5 * (d1 + d2)


def evaluate_country(
    task: torch.nn.Module,
    device: torch.device,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    tta: bool,
    watershed: bool,
    h_min: float,
    min_pad_size: int,
    backend: str,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    model = cast("torch.nn.Module", task.model)
    ds, scale = _make_dataset(country, root, split, backend)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    gsd_m = GSD_M[backend]

    metrics = MetricCollection(
        [
            JaccardIndex(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=2, ignore_index=3),
        ]
    ).to(device)
    obj_pix = [0, 0, 0]  # tp, fp, fn from connected components
    obj_ws = [0, 0, 0]  # tp, fp, fn from watershed instances
    counts = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}  # per-threshold poly counts
    matched_ious: list[float] = []
    chamfer_px: list[float] = []
    poly_deltas: list[int] = []

    for batch in tqdm(dl, desc=country, leave=False):
        image = batch["image"].to(device) / scale
        mask = batch["mask"].to(device)
        image, mask, h, w = pad_to_min(image, mask, min_size=min_pad_size)

        probs = predict_tta(model, image) if tta else predict(model, image)
        seg_pred = probs.argmax(dim=1)  # (1, H, W) 0/1/2

        # Collapse 3-class to 2-class field metrics (boundary -> background).
        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        mask_eval[mask == 3] = 3  # keep padded ignore
        metrics.update((seg_pred == 1).long(), mask_eval)

        seg_np = seg_pred.squeeze(0).cpu().numpy().astype(np.uint8)[:h, :w]
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:h, :w]

        # Object metrics from connected components of the predicted field class.
        tps, fps, fns = get_object_level_metrics(
            gt_np, (seg_np == 1).astype(np.uint8), iou_threshold=iou_threshold
        )
        obj_pix[0] += tps
        obj_pix[1] += fps
        obj_pix[2] += fns

        if watershed:
            boundary = (seg_np == 2).astype(np.uint8)
            dist = distance_transform_edt(boundary == 0).astype(np.float32)
            inst = watershed_instances(seg_np, dist, h_min=h_min)
            pred_bin = (inst > 0).astype(np.uint8)
            tps, fps, fns = get_object_level_metrics(gt_np, pred_bin, iou_threshold=iou_threshold)
            obj_ws[0] += tps
            obj_ws[1] += fps
            obj_ws[2] += fns
        else:
            pred_bin = (seg_np == 1).astype(np.uint8)

        # Polygon-level PQ / AP / count delta / boundary error.
        gt_shapes = _extract_shapes((gt_np == 1).astype(np.uint8))
        pred_shapes = _extract_shapes(pred_bin)
        poly_deltas.append(abs(len(pred_shapes) - len(gt_shapes)))
        per_t, matched_low = _match_shapes(gt_shapes, pred_shapes)
        for t, (tp, fp, fn) in per_t.items():
            counts[t][0] += tp
            counts[t][1] += fp
            counts[t][2] += fn
        matched_ious.extend(iou for _, _, iou in matched_low)
        # Chamfer over IoU>=0.5 matched pairs (rasterize each matched shape).
        for i, j, _ in matched_low:
            pm = rasterio.features.rasterize(
                [pred_shapes[j]], out_shape=pred_bin.shape, dtype=np.uint8
            )
            gm = rasterio.features.rasterize(
                [gt_shapes[i]], out_shape=pred_bin.shape, dtype=np.uint8
            )
            c = _symmetric_chamfer(_boundary_pixels(pm), _boundary_pixels(gm))
            if c is not None:
                chamfer_px.append(c)

    return _aggregate(
        metrics, obj_pix, obj_ws, counts, matched_ious, chamfer_px, poly_deltas, gsd_m, watershed
    )


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / (p + r)) if (p + r) else 0.0


def _aggregate(
    metrics: MetricCollection,
    obj_pix: list[int],
    obj_ws: list[int],
    counts: dict[float, list[int]],
    matched_ious: list[float],
    chamfer_px: list[float],
    poly_deltas: list[int],
    gsd_m: float,
    watershed: bool,
) -> dict[str, float]:
    res = metrics.compute()
    tps, fps, fns = counts[AP_IOU_THRESHOLDS[0]]
    _, _, rq = _f1(tps, fps, fns)
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    ap = float(np.mean([_f1(*counts[t])[2] for t in AP_IOU_THRESHOLDS]))
    chamfer_m = np.array(chamfer_px, dtype=np.float64) * gsd_m

    pp, pr, pf = _f1(*obj_pix)
    out = {
        "n_patches": len(poly_deltas),
        "pixel_level_iou": res["MulticlassJaccardIndex"][1].item(),
        "pixel_level_precision": res["MulticlassPrecision"][1].item(),
        "pixel_level_recall": res["MulticlassRecall"][1].item(),
        "object_pix_precision": pp,
        "object_pix_recall": pr,
        "object_pix_f1": pf,
        "pq_sq": sq,
        "pq_rq": rq,
        "pq": sq * rq,
        "ap_5_95": ap,
        "polygon_count_delta_mean": float(np.mean(poly_deltas)) if poly_deltas else 0.0,
        "polygon_count_delta_median": float(np.median(poly_deltas)) if poly_deltas else 0.0,
        "boundary_error_m_mean": float(chamfer_m.mean()) if chamfer_m.size else float("nan"),
        "boundary_error_m_p95": float(np.percentile(chamfer_m, 95))
        if chamfer_m.size
        else float("nan"),
    }
    if watershed:
        wp, wr, wf = _f1(*obj_ws)
        out["object_ws_precision"] = wp
        out["object_ws_recall"] = wr
        out["object_ws_f1"] = wf
    return out


def evaluate_checkpoint(
    ckpt: str | Path,
    out: str | Path,
    countries: tuple[str, ...],
    *,
    root: str = "data",
    split: str = "test",
    num_workers: int = 8,
    tta: bool = True,
    watershed: bool = True,
    h_min: float = 2.0,
    min_pad_size: int = 512,
    backend: str = "planet",
    gpu: int = 0,
) -> Path:
    """Evaluate ``ckpt`` over ``countries`` and write one CSV row per country."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu")
    task = load_task(ckpt, device)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for country in countries:
            print(f"=== {country} ({split}) ===")
            m = evaluate_country(
                task,
                device,
                country,
                root,
                split,
                num_workers,
                tta,
                watershed,
                h_min,
                min_pad_size,
                backend,
            )
            row = [str(ckpt), country] + [
                f"{m.get(c, float('nan')):.6f}"
                if isinstance(m.get(c), float)
                else str(m.get(c, ""))
                for c in CSV_COLUMNS[2:]
            ]
            f.write(",".join(row) + "\n")
            f.flush()
            print(
                f"  IoU={m['pixel_level_iou']:.4f} PQ={m['pq']:.4f} "
                f"obj_F1={m.get('object_ws_f1', m['object_pix_f1']):.4f}"
            )
    return out
