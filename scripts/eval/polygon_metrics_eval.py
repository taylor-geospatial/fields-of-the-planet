"""GSD-aware polygon-level metrics for field-boundary segmentation.

Per checkpoint, per country, computes:

* **PQ / SQ / RQ** (panoptic quality, IoU>=0.5 matches)
* **AP@[0.5:0.05:0.95]** (mean F1 over IoU thresholds, single-pass)
* **Polygon count delta** (|N_pred - N_gt|) per patch, mean+median
* **Boundary error (m)** symmetric chamfer * GSD on IoU>=0.5 matches,
  mean+p95

Reuses the inference/watershed pipeline from ``scripts/postprocess_eval.py``
so numbers are comparable. Outputs a single CSV row per country.

Example:
    uv run scripts/polygon_metrics_eval.py \\
        --ckpt logs/.../checkpoints/last.ckpt \\
        --out logs/polygon_metrics/foo.csv \\
        --dataset-backend planet --min-pad-size 512 \\
        --watershed --tta
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio.features
import shapely.geometry
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask

# Reuse all the inference + watershed infrastructure from postprocess_eval.
from postprocess_eval import (
    COUNTRIES,
    _pad_min32,
    _predict,
    _predict_tta,
    watershed_instances,
)
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

# IoU thresholds for AP@[0.5:0.05:0.95]
AP_IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2).tolist()

# Ground-sample distance in meters (Planet 3m, S2 10m).
GSD_M = {"planet": 3.0, "s2": 10.0}


def _extract_shapes(mask: np.ndarray) -> list[shapely.geometry.base.BaseGeometry]:
    """Connected-component shapes of value==1 in ``mask``.

    Matches the shape extraction used by ``get_object_level_metrics``.
    """
    shapes = []
    for geom, val in rasterio.features.shapes(mask.astype(np.uint8)):
        if val == 1:
            shapes.append(shapely.geometry.shape(geom))
    return shapes


AREA_BIN_LABELS = ("small", "medium", "large")


def _area_bin(area_ha: float, edges: tuple) -> str:
    """Bin a polygon by area (ha). edges=(0.5,2.0) -> small<0.5, medium 0.5-2, large>2."""
    for k, e in enumerate(edges):
        if area_ha < e:
            return AREA_BIN_LABELS[k]
    return AREA_BIN_LABELS[len(edges)]


def _match_shapes(
    gt_shapes: list,
    pred_shapes: list,
    iou_thresholds: list[float],
) -> dict:
    """Greedy-match GT vs predicted shapes once, return per-threshold tps/fps/fns
    plus the matched IoUs and pair indices at the lowest threshold.

    For a fixed greedy strategy (the same one ``get_object_level_metrics``
    uses — first prediction with IoU>thresh wins), tps/fps/fns at every
    threshold can be derived from a single pairwise IoU pass. This keeps
    AP@[0.5:0.95] cheap.
    """
    n_gt, n_pred = len(gt_shapes), len(pred_shapes)
    # Precompute IoUs for all intersecting pairs.
    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, g in enumerate(gt_shapes):
        for j, p in enumerate(pred_shapes):
            if not g.intersects(p):
                continue
            inter = g.intersection(p).area
            if inter <= 0:
                continue
            union = g.area + p.area - inter
            if union <= 0:
                continue
            ious[i, j] = inter / union

    per_t = {}
    pairs_per_t: dict[float, list] = {}  # matched (i,j,iou) at each threshold
    for t in iou_thresholds:
        matched_j: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for i in range(n_gt):
            for j in range(n_pred):
                if j in matched_j:
                    continue
                iou = float(ious[i, j])
                if iou > t:
                    matched_j.add(j)
                    pairs.append((i, j, iou))
                    break
        per_t[t] = (len(pairs), n_pred - len(matched_j), n_gt - len(pairs))
        pairs_per_t[t] = pairs
    return {
        "per_t": per_t,
        "matched_pairs_low": pairs_per_t[iou_thresholds[0]],  # SQ + chamfer
        "pairs_per_t": pairs_per_t,
    }


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Boolean array of 1-pixel-thick boundary of ``mask`` (uint8 0/1)."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    # Boundary = pixels in mask whose distance-to-background is < 1.5
    # (4-connected boundary). Faster than morphological gradient.
    dt = distance_transform_edt(mask)
    return (mask.astype(bool)) & (dt < 1.5)


def _symmetric_chamfer(pred_bd: np.ndarray, gt_bd: np.ndarray) -> float | None:
    """Mean symmetric chamfer (pixels) between two boundary masks."""
    if not pred_bd.any() or not gt_bd.any():
        return None
    dt_pred = distance_transform_edt(~pred_bd)
    dt_gt = distance_transform_edt(~gt_bd)
    d1 = float(dt_pred[gt_bd].mean())
    d2 = float(dt_gt[pred_bd].mean())
    return 0.5 * (d1 + d2)


def evaluate_country(
    task,
    model,
    device,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    use_tta: bool,
    use_watershed: bool,
    h_min: float,
    sdf_clip: float,
    min_pad_size: int,
    pad_mode: str,
    dataset_backend: str,
    s2_data_scale: float,
    upsample_to: int | None,
    area_edges: tuple | None = None,
    pixel_area_ha: float = 0.0,
    bin_stats: dict | None = None,
) -> dict[str, float]:
    if dataset_backend == "s2":
        from ftw_tools.training.datasets import FTW

        ds = FTW(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
            temporal_options="stacked",
        )
        scale = float(s2_data_scale)
    else:
        ds = FTWPlanet(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
        )
        scale = PLANET_SR_SCALE
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    gsd_m = GSD_M[dataset_backend]

    # Per-threshold counts (for AP and PQ).
    counts = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}  # tps, fps, fns
    matched_ious_05: list[float] = []  # for SQ
    chamfer_pixels: list[float] = []  # for boundary error
    polygon_deltas: list[int] = []
    n_pred_per_patch: list[int] = []
    n_gt_per_patch: list[int] = []
    n_patches = 0

    for batch in tqdm(dl, desc=country, leave=False):
        image = batch["image"].to(device) / scale
        mask = batch["mask"].to(device)
        if upsample_to is not None:
            # Resized-S2 control: bilinear-upsample image, nearest-upsample mask
            # 256->upsample_to to match the upsampled-S2 training resolution.
            image = torch.nn.functional.interpolate(
                image, size=(upsample_to, upsample_to), mode="bilinear", align_corners=False
            )
            mask = (
                torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(), size=(upsample_to, upsample_to), mode="nearest"
                )
                .squeeze(1)
                .to(mask.dtype)
            )
        image, mask, H, W = _pad_min32(image, mask, min_size=min_pad_size, pad_mode=pad_mode)

        if use_tta:
            probs, sdf = _predict_tta(task, model, image, sdf_clip)
        else:
            probs, sdf = _predict(task, model, image, sdf_clip)
        seg_pred = probs.argmax(dim=1)  # (1,H,W) 0/1/2

        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        mask_eval[mask == 3] = 3

        seg_np = seg_pred.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        if use_watershed:
            if sdf is not None:
                dist = sdf.squeeze(0).cpu().numpy().astype(np.float32)[:H, :W]
            else:
                boundary_np = (seg_np == 2).astype(np.uint8)
                dist = distance_transform_edt(boundary_np == 0).astype(np.float32)
            inst_pred = watershed_instances(seg_np, dist, h_min=h_min, field_class=1)
            pred_bin = (inst_pred > 0).astype(np.uint8)
        else:
            pred_bin = (seg_np == 1).astype(np.uint8)

        gt_bin = (gt_np == 1).astype(np.uint8)

        gt_shapes = _extract_shapes(gt_bin)
        pred_shapes = _extract_shapes(pred_bin)

        n_patches += 1
        n_pred_per_patch.append(len(pred_shapes))
        n_gt_per_patch.append(len(gt_shapes))
        polygon_deltas.append(abs(len(pred_shapes) - len(gt_shapes)))

        m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)
        for t, (tps, fps, fns) in m["per_t"].items():
            counts[t][0] += tps
            counts[t][1] += fps
            counts[t][2] += fns

        # SQ + chamfer use the IoU>=0.5 matches.
        for i, j, iou in m["matched_pairs_low"]:
            matched_ious_05.append(iou)
            # Rasterize each matched shape into a tight bbox and chamfer.
            pred_mask = rasterio.features.rasterize(
                [pred_shapes[j]], out_shape=pred_bin.shape, dtype=np.uint8
            )
            gt_mask = rasterio.features.rasterize(
                [gt_shapes[i]], out_shape=gt_bin.shape, dtype=np.uint8
            )
            pb = _boundary_pixels(pred_mask)
            gb = _boundary_pixels(gt_mask)
            c = _symmetric_chamfer(pb, gb)
            if c is not None:
                chamfer_pixels.append(c)

        # ---- per-area-bin x per-IoU-threshold accumulation ----
        # GT binned by GT polygon area (ha); predicted FPs by predicted area.
        if bin_stats is not None:
            gt_bins = [_area_bin(g.area * pixel_area_ha, area_edges) for g in gt_shapes]
            pred_bins = [_area_bin(p.area * pixel_area_ha, area_edges) for p in pred_shapes]
            for t in AP_IOU_THRESHOLDS:
                matched_gt = {i: iou for i, j, iou in m["pairs_per_t"][t]}
                matched_pred = {j for _, j, _ in m["pairs_per_t"][t]}
                for i, b in enumerate(gt_bins):
                    if i in matched_gt:
                        bin_stats[b][t]["tp"] += 1
                        bin_stats[b][t]["ious"].append(matched_gt[i])
                    else:
                        bin_stats[b][t]["fn"] += 1
                for j, b in enumerate(pred_bins):
                    if j not in matched_pred:
                        bin_stats[b][t]["fp"] += 1

    # ---- Aggregate ----
    def _f1(tps: int, fps: int, fns: int) -> float:
        p = tps / max(tps + fps, 1)
        r = tps / max(tps + fns, 1)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    # PQ at IoU>=0.5
    t05 = AP_IOU_THRESHOLDS[0]
    tps, fps, fns = counts[t05]
    rq = _f1(tps, fps, fns)
    sq = float(np.mean(matched_ious_05)) if matched_ious_05 else 0.0
    pq = sq * rq
    ap = float(np.mean([_f1(*counts[t]) for t in AP_IOU_THRESHOLDS]))

    chamfer_m = (
        np.array(chamfer_pixels, dtype=np.float64) * gsd_m
        if chamfer_pixels
        else np.array([], dtype=np.float64)
    )

    return {
        "n_patches": n_patches,
        "pq_sq": sq,
        "pq_rq": rq,
        "pq": pq,
        "ap_5_95": ap,
        "n_pred_mean": float(np.mean(n_pred_per_patch)) if n_pred_per_patch else 0.0,
        "n_gt_mean": float(np.mean(n_gt_per_patch)) if n_gt_per_patch else 0.0,
        "polygon_count_delta_mean": (float(np.mean(polygon_deltas)) if polygon_deltas else 0.0),
        "polygon_count_delta_median": (float(np.median(polygon_deltas)) if polygon_deltas else 0.0),
        "boundary_error_m_mean": float(chamfer_m.mean()) if chamfer_m.size else float("nan"),
        "boundary_error_m_p95": (
            float(np.percentile(chamfer_m, 95)) if chamfer_m.size else float("nan")
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--countries", nargs="*", default=None)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--watershed", action="store_true")
    p.add_argument("--h-min", type=float, default=2.0)
    p.add_argument("--sdf-clip", type=float, default=20.0)
    p.add_argument("--min-pad-size", type=int, default=0)
    p.add_argument("--pad-mode", type=str, default="zero", choices=["zero", "replicate"])
    p.add_argument("--dataset-backend", type=str, default="planet", choices=["planet", "s2"])
    p.add_argument("--s2-data-scale", type=float, default=3000.0)
    p.add_argument(
        "--upsample-to",
        type=int,
        default=None,
        help="Resized-S2 control: upsample image (bilinear) + mask (nearest) "
        "256->N before padding. Use with --dataset-backend s2 and N==min-pad-size.",
    )
    p.add_argument(
        "--area-bins",
        type=str,
        default=None,
        help="Comma-separated ha edges, e.g. '0.5,2' -> small<0.5 / medium 0.5-2 / "
        "large>2 by GT polygon area. Pooled (micro) PQ/RQ/SQ per bin are written to "
        "<out>.bins.csv. Requires --pixel-size-m.",
    )
    p.add_argument(
        "--pixel-size-m",
        type=float,
        default=None,
        help="Physical pixel size (m) of the eval grid, for area->ha conversion: "
        "planet native 3, s2 native 10, s2 upsample-512 5.",
    )
    args = p.parse_args()
    area_edges = tuple(float(x) for x in args.area_bins.split(",")) if args.area_bins else None
    pixel_area_ha = (args.pixel_size_m**2) / 1e4 if args.pixel_size_m else 0.0
    bin_stats = (
        {
            b: {t: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for t in AP_IOU_THRESHOLDS}
            for b in AREA_BIN_LABELS
        }
        if area_edges
        else None
    )
    if area_edges and not args.pixel_size_m:
        p.error("--area-bins requires --pixel-size-m")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(
        f"device={device} ckpt={args.ckpt} backend={args.dataset_backend} "
        f"tta={args.tta} watershed={args.watershed}"
    )

    tic = time.time()
    task = None
    try:
        from ftw_planet.trainers import FrameFieldSegTask

        task = FrameFieldSegTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
        print("loaded as FrameFieldSegTask")
    except Exception:
        pass
    if task is None:
        try:
            from ftw_planet.trainers import SDFSegTask

            task = SDFSegTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
            print("loaded as SDFSegTask")
        except Exception:
            task = CustomSemanticSegmentationTask.load_from_checkpoint(
                str(args.ckpt), map_location="cpu"
            )
            print("loaded as CustomSemanticSegmentationTask")
    task = task.eval().to(device)
    model = task.model
    print(f"loaded model in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "train_checkpoint",
        "country",
        "n_patches",
        "pq_sq",
        "pq_rq",
        "pq",
        "ap_5_95",
        "n_pred_mean",
        "n_gt_mean",
        "polygon_count_delta_mean",
        "polygon_count_delta_median",
        "boundary_error_m_mean",
        "boundary_error_m_p95",
    ]
    if not args.out.exists():
        with args.out.open("w") as f:
            f.write(",".join(cols) + "\n")

    countries = args.countries or COUNTRIES
    for country in countries:
        print(f"=== {country} ({args.split}) ===")
        try:
            m = evaluate_country(
                task,
                model,
                device,
                country,
                args.root,
                args.split,
                args.num_workers,
                args.tta,
                args.watershed,
                args.h_min,
                args.sdf_clip,
                min_pad_size=args.min_pad_size,
                pad_mode=args.pad_mode,
                dataset_backend=args.dataset_backend,
                s2_data_scale=args.s2_data_scale,
                upsample_to=args.upsample_to,
                area_edges=area_edges,
                pixel_area_ha=pixel_area_ha,
                bin_stats=bin_stats,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  skip {country}: {e}")
            continue

        row = [
            str(args.ckpt),
            country,
            str(m["n_patches"]),
            f"{m['pq_sq']:.6f}",
            f"{m['pq_rq']:.6f}",
            f"{m['pq']:.6f}",
            f"{m['ap_5_95']:.6f}",
            f"{m['n_pred_mean']:.4f}",
            f"{m['n_gt_mean']:.4f}",
            f"{m['polygon_count_delta_mean']:.4f}",
            f"{m['polygon_count_delta_median']:.4f}",
            f"{m['boundary_error_m_mean']:.4f}",
            f"{m['boundary_error_m_p95']:.4f}",
        ]
        with args.out.open("a") as f:
            f.write(",".join(row) + "\n")
        print(
            f"  PQ={m['pq']:.4f} (SQ={m['pq_sq']:.3f}, RQ={m['pq_rq']:.3f}) "
            f"AP={m['ap_5_95']:.4f} bnd_err={m['boundary_error_m_mean']:.2f}m"
        )

    if bin_stats is not None:
        # Pooled (micro) PQ/SQ/RQ per area bin, across area bins AND IoU thresholds.
        t05 = AP_IOU_THRESHOLDS[0]
        t75 = min(AP_IOU_THRESHOLDS, key=lambda x: abs(x - 0.75))

        def _f1(tp: int, fp: int, fn: int) -> float:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        all_bin = {t: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for t in AP_IOU_THRESHOLDS}
        for b in AREA_BIN_LABELS:
            for t in AP_IOU_THRESHOLDS:
                for k in ("tp", "fp", "fn"):
                    all_bin[t][k] += bin_stats[b][t][k]
                all_bin[t]["ious"] += bin_stats[b][t]["ious"]

        bins_out = Path(str(args.out) + ".bins.csv")
        with bins_out.open("w") as f:
            f.write("area_bins,bin,n_gt,n_pred,pq,sq,rq_50,f1_75,ap_5_95\n")
            for label, st in [*((b, bin_stats[b]) for b in AREA_BIN_LABELS), ("all", all_bin)]:
                rq = _f1(st[t05]["tp"], st[t05]["fp"], st[t05]["fn"])
                f1_75 = _f1(st[t75]["tp"], st[t75]["fp"], st[t75]["fn"])
                ap = float(np.mean([_f1(st[t]["tp"], st[t]["fp"], st[t]["fn"]) for t in AP_IOU_THRESHOLDS]))
                sq = float(np.mean(st[t05]["ious"])) if st[t05]["ious"] else 0.0
                n_gt = st[t05]["tp"] + st[t05]["fn"]
                n_pred = st[t05]["tp"] + st[t05]["fp"]
                f.write(f"{args.area_bins},{label},{n_gt},{n_pred},{sq * rq:.4f},{sq:.4f},{rq:.4f},{f1_75:.4f},{ap:.4f}\n")
                print(
                    f"  [{label:7s}] n_gt={n_gt:6d} PQ={sq * rq * 100:5.1f} SQ={sq * 100:5.1f} "
                    f"RQ@.5={rq * 100:5.1f} F1@.75={f1_75 * 100:5.1f} AP={ap * 100:5.1f}"
                )
        print(f"wrote {bins_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
