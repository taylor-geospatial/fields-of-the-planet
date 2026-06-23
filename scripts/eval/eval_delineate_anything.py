"""Evaluate off-the-shelf DelineateAnything (YOLO11x-seg) and DelineateAnything-S
(YOLO11n-seg) as zero-shot polygon-level baselines on FTP (PlanetScope) and FTW
(Sentinel-2).

The released DelineateAnything models are trained on a mix of RGB uint8
satellite imagery (mostly European; no FTW countries) and output field
instance masks directly. We mirror ``scripts/eval/polygon_metrics_eval.py``
so the numbers line up with our PRUE+ rows:

* Take the season-B window (4-band SR for Planet; 4-band for S2).
* RGB uint8 via clip(refl / ceiling, 0, 1) * 255.
    - Planet PSScene band order = (Blue, Green, Red, NIR) -> RGB = [2, 1, 0].
    - FTW S2 band order      = (Red, Green, Blue, NIR)  -> RGB = [0, 1, 2].
* Run ultralytics inference; per-detection masks -> shapely polygons -> the
  same _match_shapes / symmetric-chamfer / area-bin code path as our model.

No watershed (DelineateAnything outputs instances directly). Zero-shot: no
Planet/FTW fine-tuning.

Example:
    uv run scripts/eval/eval_delineate_anything.py \\
        --dataset-backend planet --root data \\
        --out logs/polygon_metrics/delineate_x_planet.csv \\
        --area-bins 0.5,2 --pixel-size-m 3 \\
        --conf 0.005 --iou 0.5 --imgsz 1024
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio.features
import shapely.geometry
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from polygon_metrics_eval import (
    AP_IOU_THRESHOLDS,
    AREA_BIN_LABELS,
    GSD_M,
    _area_bin,
    _boundary_pixels,
    _cap_pred_to_gsd,
    _eval_grid,
    _extract_shapes,
    _match_shapes,
    _symmetric_chamfer,
    _true_gt_utm,
)
from postprocess_eval import COUNTRIES
from shapely.ops import unary_union
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import (
    YOLO,  # ultralytics is an optional dep not installed in base CI env
)

from ftw_planet.datasets import FTWPlanet

# Default per-band reflectance-DN ceiling mapping uint16 SR -> uint8 RGB.
# Matches the hero figure's `value / 3000 clip [0, 1]` rescaling.
RGB_REFL_CEILING = 3000.0

# Window-B RGB channel order within the 8-channel (B-first, A-second) stack.
# Planet PSScene = (B,G,R,NIR); FTW S2 = (R,G,B,NIR).
RGB_CHANNELS = {"planet": [2, 1, 0], "s2": [0, 1, 2]}


def window_b_to_rgb_uint8(window_8ch: np.ndarray, ceiling: float, backend: str) -> np.ndarray:
    """Pull window B's RGB out of the 8-channel (B,A) stack and rescale to uint8.

    Input  : (8, H, W) float DN (window B bands 1..4, then window A 1..4).
    Output : (H, W, 3) uint8 in R,G,B order, suitable for YOLO.
    """
    win_b = window_8ch[:4]  # (4, H, W) for window B
    rgb = win_b[RGB_CHANNELS[backend]]  # -> (R, G, B)
    rgb = np.clip(rgb / ceiling, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return np.transpose(rgb_u8, (1, 2, 0))


def _make_dataset(backend: str, root: str, country: str, split: str):
    if backend == "s2":
        from ftw_tools.training.datasets import FTW

        return FTW(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
            temporal_options="stacked",
        )
    return FTWPlanet(
        root=root, countries=[country], split=split, transforms=None, load_boundaries=True
    )


def evaluate_country_yolo(
    yolo: YOLO,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    conf: float,
    iou_thresh: float,
    imgsz: int,
    device: torch.device,
    backend: str,
    ceiling: float,
    bin_stats: dict | None,
    pixel_area_ha: float,
    area_edges: tuple | None,
    rgb_sample_path: Path | None = None,
    gt_polygons_root: str | None = None,
    planet_window: str = "a",
    score_gsd_m: float | None = None,
) -> dict[str, float]:
    ds = _make_dataset(backend, root, country, split)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
    gsd_m = GSD_M[backend]
    # True-GT scoring: match against the true FTW polygons in metric UTM space
    # (prediction capped to the sensor's native ground resolution), mirroring
    # polygon_metrics_eval so DA rows are comparable to the PRUE+ rows.
    gsd_mode = score_gsd_m is not None and gt_polygons_root is not None

    counts = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}
    matched_ious_05: list[float] = []
    chamfer_pixels: list[float] = []
    polygon_deltas: list[int] = []
    n_pred_per_patch: list[int] = []
    n_gt_per_patch: list[int] = []
    pix_inter = 0
    pix_union = 0
    n_patches = 0

    for idx, batch in enumerate(tqdm(dl, desc=country, leave=False)):
        image = batch["image"][0].cpu().numpy().astype(np.float32)  # (8, H, W) raw DN
        mask = batch["mask"][0].cpu().numpy().astype(np.int64)
        H, W = mask.shape

        rgb_u8 = window_b_to_rgb_uint8(image, ceiling, backend)
        if rgb_sample_path is not None and n_patches == 0:
            Image.fromarray(rgb_u8).save(rgb_sample_path)
            print(f"  wrote RGB sanity sample {rgb_sample_path} (mean={rgb_u8.mean():.1f})")

        # ultralytics treats a numpy source as BGR and flips it to RGB in
        # preprocess; pass BGR so the model receives true RGB (else R/B swap).
        results = yolo.predict(
            source=rgb_u8[..., ::-1],
            imgsz=imgsz,
            conf=conf,
            iou=iou_thresh,
            device=str(device) if device.type != "cpu" else "cpu",
            verbose=False,
            retina_masks=True,
            max_det=2000,
            half=device.type == "cuda",
        )
        r = results[0]

        gt_eval = mask.copy()
        gt_eval[gt_eval == 2] = 0  # collapse boundary class
        gt_eval[mask == 3] = 3
        gt_bin = (gt_eval == 1).astype(np.uint8)
        gt_shapes = _extract_shapes(gt_bin)

        pred_shapes: list[shapely.geometry.base.BaseGeometry] = []
        if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
            masks = r.masks.data.cpu().numpy().astype(np.uint8)  # (N, h, w)
            for k in range(masks.shape[0]):
                m_k = masks[k]
                if m_k.shape != (H, W):
                    m_k = m_k[:H, :W]
                    if m_k.shape != (H, W):
                        pad_h, pad_w = H - m_k.shape[0], W - m_k.shape[1]
                        m_k = np.pad(m_k, ((0, max(0, pad_h)), (0, max(0, pad_w))))
                if m_k.sum() == 0:
                    continue
                # One YOLO detection = one predicted instance: union its blobs
                # into a single (multi)polygon rather than splitting a multi-blob
                # mask into several instances (which would over-count and give
                # the instance segmenter extra matching chances).
                geoms = [
                    shapely.geometry.shape(g)
                    for g, val in rasterio.features.shapes(m_k)
                    if val == 1
                ]
                geoms = [g for g in geoms if g.area > 0]
                if not geoms:
                    continue
                pred_shapes.append(geoms[0] if len(geoms) == 1 else unary_union(geoms))

        # Object-level scoring shapes. In GSD mode, match the TRUE FTW polygons
        # (UTM) against the prediction capped to the sensor's native ground
        # resolution; otherwise use the rasterized-GT pixel shapes.
        gt_true_areas = None
        obj_pixel_area_ha = pixel_area_ha
        if gsd_mode:
            patch_id, eval_crs, eval_tr, _ = _eval_grid(
                ds, idx, country, backend, root, planet_window, None
            )
            gt_obj, gt_true_areas, utm_crs = _true_gt_utm(gt_polygons_root, country, patch_id)
            pred_obj = _cap_pred_to_gsd(pred_shapes, eval_tr, eval_crs, utm_crs, score_gsd_m)
            obj_pixel_area_ha = 1e-4  # UTM m^2 -> ha
        else:
            gt_obj, pred_obj = gt_shapes, pred_shapes

        n_patches += 1
        n_pred_per_patch.append(len(pred_obj))
        n_gt_per_patch.append(len(gt_obj))
        polygon_deltas.append(abs(len(pred_obj) - len(gt_obj)))

        m = _match_shapes(gt_obj, pred_obj, AP_IOU_THRESHOLDS)
        for t, (tps, fps, fns) in m["per_t"].items():
            counts[t][0] += tps
            counts[t][1] += fps
            counts[t][2] += fns

        # Reconstruct flattened pred mask for chamfer + pixel IoU.
        pred_bin = np.zeros_like(gt_bin)
        for s in pred_shapes:
            try:
                pred_bin = np.maximum(
                    pred_bin,
                    rasterio.features.rasterize([s], out_shape=gt_bin.shape, dtype=np.uint8),
                )
            except ValueError:
                continue

        # Pixel IoU (field-interior class), micro-pooled over the country.
        valid = mask != 3  # exclude ignore-padded pixels
        g = (gt_bin > 0) & valid
        p = (pred_bin > 0) & valid
        pix_inter += int((g & p).sum())
        pix_union += int((g | p).sum())

        for i, j, iou in m["matched_pairs_low"]:
            matched_ious_05.append(iou)
            if gsd_mode:
                # Object IoUs (hence SQ) live in UTM; the boundary chamfer for the
                # table comes from the rasterized-GT native-grid run, so skip it
                # here. SQ is scale-free and still accumulates above.
                continue
            pred_mask = rasterio.features.rasterize(
                [pred_shapes[j]], out_shape=pred_bin.shape, dtype=np.uint8
            )
            gt_mask = rasterio.features.rasterize(
                [gt_shapes[i]], out_shape=gt_bin.shape, dtype=np.uint8
            )
            c = _symmetric_chamfer(_boundary_pixels(pred_mask), _boundary_pixels(gt_mask))
            if c is not None:
                chamfer_pixels.append(c)

        # Per-area-bin x per-IoU-threshold accumulation (mirrors polygon_metrics_eval).
        if bin_stats is not None:
            if gt_true_areas is not None:
                gt_bins = [_area_bin(a, area_edges) for a in gt_true_areas]
            else:
                gt_bins = [_area_bin(g.area * pixel_area_ha, area_edges) for g in gt_obj]
            pred_bins = [_area_bin(p.area * obj_pixel_area_ha, area_edges) for p in pred_obj]
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

    def _f1(tps: int, fps: int, fns: int) -> float:
        p = tps / max(tps + fps, 1)
        r = tps / max(tps + fns, 1)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

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
        "pixel_level_iou": (pix_inter / pix_union) if pix_union else 0.0,
        "n_pred_mean": float(np.mean(n_pred_per_patch)) if n_pred_per_patch else 0.0,
        "n_gt_mean": float(np.mean(n_gt_per_patch)) if n_gt_per_patch else 0.0,
        "polygon_count_delta_mean": float(np.mean(polygon_deltas)) if polygon_deltas else 0.0,
        "polygon_count_delta_median": float(np.median(polygon_deltas)) if polygon_deltas else 0.0,
        "boundary_error_m_mean": float(chamfer_m.mean()) if chamfer_m.size else float("nan"),
        "boundary_error_m_p95": (
            float(np.percentile(chamfer_m, 95)) if chamfer_m.size else float("nan")
        ),
    }


def _write_bins_csv(out: Path, bin_stats: dict, area_tag: str) -> None:
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

    bins_out = Path(str(out) + ".bins.csv")
    with bins_out.open("w") as f:
        f.write("area_bins,bin,n_gt,n_pred,pq,sq,rq_50,f1_75,ap_5_95\n")
        for label, st in [*((b, bin_stats[b]) for b in AREA_BIN_LABELS), ("all", all_bin)]:
            rq = _f1(st[t05]["tp"], st[t05]["fp"], st[t05]["fn"])
            f1_75 = _f1(st[t75]["tp"], st[t75]["fp"], st[t75]["fn"])
            ap = float(
                np.mean([_f1(st[t]["tp"], st[t]["fp"], st[t]["fn"]) for t in AP_IOU_THRESHOLDS])
            )
            sq = float(np.mean(st[t05]["ious"])) if st[t05]["ious"] else 0.0
            n_gt = st[t05]["tp"] + st[t05]["fn"]
            n_pred = st[t05]["tp"] + st[t05]["fp"]
            f.write(
                f"{area_tag},{label},{n_gt},{n_pred},{sq * rq:.4f},{sq:.4f},{rq:.4f},{f1_75:.4f},{ap:.4f}\n"
            )
    print(f"wrote {bins_out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data", type=str, help="planet: data; s2: data/ftw")
    p.add_argument("--dataset-backend", default="planet", choices=["planet", "s2"])
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--countries", nargs="*", default=None)
    p.add_argument("--conf", type=float, default=0.005)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--imgsz", type=int, default=512
    )  # DelineateAnything ckpts trained at 512 (square)
    p.add_argument("--rgb-ceiling", type=float, default=RGB_REFL_CEILING)
    p.add_argument(
        "--area-bins", type=str, default=None, help="e.g. 0.5,2 (small/medium/large edges)"
    )
    p.add_argument("--pixel-size-m", type=float, default=None, help="GSD for area; planet 3, s2 10")
    p.add_argument("--save-rgb-sample", type=Path, default=None, help="save first-patch RGB PNG")
    p.add_argument("--hf-repo", type=str, default="torchgeo/delineate-anything")
    p.add_argument("--hf-file", type=str, default="delineate_anything_rgb_yolo11x-88ede029.pt")
    p.add_argument(
        "--gt-polygons-root",
        type=str,
        default=None,
        help="Root of per-patch true FTW polygons (UTM parquet, e.g. data/ftw_polygons_clipped); "
        "enables true-GT scoring comparable to the PRUE+ rows.",
    )
    p.add_argument(
        "--score-gsd-m",
        type=float,
        default=None,
        help="Cap predictions to this ground resolution before matching (planet 3, s2 10); "
        "requires --gt-polygons-root. Object metrics + size bins use true polygons in UTM.",
    )
    p.add_argument(
        "--planet-window",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Planet window whose grid defines the eval CRS/transform (geometry is identical "
        "across windows for a patch).",
    )
    args = p.parse_args()

    area_edges = tuple(float(x) for x in args.area_bins.split(",")) if args.area_bins else None
    pixel_area_ha = (args.pixel_size_m**2) / 1e4 if args.pixel_size_m else 0.0
    if area_edges and not args.pixel_size_m:
        p.error("--area-bins requires --pixel-size-m")
    if args.score_gsd_m is not None and args.gt_polygons_root is None:
        p.error("--score-gsd-m requires --gt-polygons-root")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(
        f"device={device} backend={args.dataset_backend} conf={args.conf} iou={args.iou} "
        f"imgsz={args.imgsz} ceiling={args.rgb_ceiling} split={args.split} ckpt={args.hf_file}"
    )

    tic = time.time()
    ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file)
    yolo = YOLO(ckpt_path)
    print(f"loaded YOLO ckpt in {time.time() - tic:.1f}s: {ckpt_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "model",
        "country",
        "n_patches",
        "pq_sq",
        "pq_rq",
        "pq",
        "ap_5_95",
        "pixel_level_iou",
        "n_pred_mean",
        "n_gt_mean",
        "polygon_count_delta_mean",
        "polygon_count_delta_median",
        "boundary_error_m_mean",
        "boundary_error_m_p95",
        "conf",
        "iou",
        "imgsz",
    ]
    if not args.out.exists():
        with args.out.open("w") as f:
            f.write(",".join(cols) + "\n")

    model_stem = Path(args.hf_file).stem.split("-")[0]  # delineate_anything[_s]_rgb_yolo11x
    tag = f"{model_stem}@{args.dataset_backend}_conf{args.conf}_iou{args.iou}_sz{args.imgsz}"

    bin_stats = (
        {
            b: {t: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for t in AP_IOU_THRESHOLDS}
            for b in AREA_BIN_LABELS
        }
        if area_edges
        else None
    )
    bin_stats = dict(bin_stats) if bin_stats is not None else None

    countries = args.countries or COUNTRIES
    for ci, country in enumerate(countries):
        print(f"=== {country} ({args.split}) ===")
        try:
            m = evaluate_country_yolo(
                yolo,
                country,
                args.root,
                args.split,
                args.num_workers,
                args.conf,
                args.iou,
                args.imgsz,
                device,
                args.dataset_backend,
                args.rgb_ceiling,
                bin_stats,
                pixel_area_ha,
                area_edges,
                rgb_sample_path=(args.save_rgb_sample if ci == 0 else None),
                gt_polygons_root=args.gt_polygons_root,
                planet_window=args.planet_window,
                score_gsd_m=args.score_gsd_m,
            )
        except FileNotFoundError as e:
            print(f"  skip {country}: {e}")
            continue

        row = [
            tag,
            country,
            str(m["n_patches"]),
            f"{m['pq_sq']:.6f}",
            f"{m['pq_rq']:.6f}",
            f"{m['pq']:.6f}",
            f"{m['ap_5_95']:.6f}",
            f"{m['pixel_level_iou']:.6f}",
            f"{m['n_pred_mean']:.4f}",
            f"{m['n_gt_mean']:.4f}",
            f"{m['polygon_count_delta_mean']:.4f}",
            f"{m['polygon_count_delta_median']:.4f}",
            f"{m['boundary_error_m_mean']:.4f}",
            f"{m['boundary_error_m_p95']:.4f}",
            f"{args.conf}",
            f"{args.iou}",
            f"{args.imgsz}",
        ]
        with args.out.open("a") as f:
            f.write(",".join(row) + "\n")
        print(
            f"  pq={m['pq']:.3f} ap={m['ap_5_95']:.3f} pixIoU={m['pixel_level_iou']:.3f} "
            f"n_pred/n_gt={m['n_pred_mean']:.1f}/{m['n_gt_mean']:.1f} bnd_m={m['boundary_error_m_mean']:.1f}"
        )

    if bin_stats is not None:
        _write_bins_csv(args.out, bin_stats, args.area_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
