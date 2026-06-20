"""Evaluate the off-the-shelf DelineateAnything (YOLO11x-seg) checkpoint on
FTW-HD as a zero-shot polygon-level baseline.

The released DelineateAnything model is trained on a mix of RGB uint8
satellite imagery (mostly European; no FTW countries) and outputs field
instance masks directly. We mirror ``scripts/polygon_metrics_eval.py``
so the numbers line up with our PRUE-HD rows:

* Take the season-B PlanetScope window (4-band uint16 SR).
* RGB = bands [R, G, B] = source band indices [2, 1, 0].
* Scale uint16 -> uint8 via clip(refl / 3000, 0, 1) * 255 (matches the
  paper's visualization rescaling).
* Run ultralytics inference; per-detection masks are turned into
  shapely polygons and fed into the same _match_shapes /
  symmetric-chamfer code path as our model.

No watershed step (DelineateAnything outputs instances directly). No
training-window aug; this is a zero-shot eval. Honest framing:
EU-trained instance model, no Planet/FTW fine-tuning.

Example:
    uv run scripts/eval_delineate_anything.py \\
        --out logs/polygon_metrics/delineate_anything.csv \\
        --conf 0.25 --iou 0.5 --imgsz 1024
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio.features
import shapely.geometry
import torch
from huggingface_hub import hf_hub_download
from polygon_metrics_eval import (
    AP_IOU_THRESHOLDS,
    GSD_M,
    _boundary_pixels,
    _extract_shapes,
    _match_shapes,
    _symmetric_chamfer,
)
from postprocess_eval import COUNTRIES
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import (
    YOLO,  # ultralytics is an optional dep not installed in base CI env
)

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

# Per-band reflectance ceiling we use to map uint16 SR -> uint8 RGB.
# Matches the hero figure's `value / 3000 clip [0, 1]` rescaling.
RGB_REFL_CEILING = 3000.0


def planet_window_to_rgb_uint8(window_8ch: np.ndarray, ceiling: float) -> np.ndarray:
    """Pull window B's RGB out of the 8-channel (B,A) stack and rescale.

    Input  : (8, H, W) float tensor (window B bands 1..4, then window A 1..4)
             ordered (B,G,R,NIR) per Planet PSScene.
    Output : (H, W, 3) uint8 in R,G,B order, suitable for YOLO.

    The 8-ch stack is window-B-first by the dataset's default ordering. We
    use only window B and drop NIR; DelineateAnything is RGB-only.
    """
    win_b = window_8ch[:4]  # (4, H, W) for window B
    # Planet PSScene band order = (Blue, Green, Red, NIR).
    rgb = win_b[[2, 1, 0]]  # -> (R, G, B)
    rgb = np.clip(rgb / ceiling, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return np.transpose(rgb_u8, (1, 2, 0))


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
) -> dict[str, float]:
    ds = FTWPlanet(
        root=root,
        countries=[country],
        split=split,
        transforms=None,
        load_boundaries=True,
    )
    scale = PLANET_SR_SCALE  # noqa: F841  -- shows we read uint16 -> /scale = reflectance
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)

    gsd_m = GSD_M["planet"]

    counts = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}
    matched_ious_05: list[float] = []
    chamfer_pixels: list[float] = []
    polygon_deltas: list[int] = []
    n_pred_per_patch: list[int] = []
    n_gt_per_patch: list[int] = []
    n_patches = 0

    for batch in tqdm(dl, desc=country, leave=False):
        image = batch["image"][0].cpu().numpy().astype(np.float32)  # (8, H, W) raw DN
        mask = batch["mask"][0].cpu().numpy().astype(np.int64)
        H, W = mask.shape

        rgb_u8 = planet_window_to_rgb_uint8(image, RGB_REFL_CEILING)

        results = yolo.predict(
            source=rgb_u8,
            imgsz=imgsz,
            conf=conf,
            iou=iou_thresh,
            device=str(device) if device.type != "cpu" else "cpu",
            verbose=False,
            retina_masks=True,  # masks at full input resolution
            max_det=2000,  # Planet patches in Cambodia/Vietnam exceed YOLO's 300 default
            half=True,  # matches DelineateAnything reference config (`use_half: true`)
        )
        r = results[0]

        gt_eval = mask.copy()
        gt_eval[gt_eval == 2] = 0  # collapse boundary class
        gt_eval[mask == 3] = 3
        gt_bin = (gt_eval == 1).astype(np.uint8)
        gt_shapes = _extract_shapes(gt_bin)

        pred_shapes: list[shapely.geometry.base.BaseGeometry] = []
        if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
            masks = (
                r.masks.data.cpu().numpy().astype(np.uint8)
            )  # YOLO Results.masks.data is always Tensor here; guarded by None-check above  # (N, h, w)
            # Per-instance polygons.
            for k in range(masks.shape[0]):
                m_k = masks[k]
                if m_k.shape != (H, W):
                    # retina_masks=True usually returns at input H/W; if YOLO
                    # internally padded to imgsz, take a top-left crop.
                    m_k = m_k[:H, :W]
                    if m_k.shape != (H, W):
                        # pad to (H, W) if smaller (rare)
                        pad_h = H - m_k.shape[0]
                        pad_w = W - m_k.shape[1]
                        m_k = np.pad(m_k, ((0, max(0, pad_h)), (0, max(0, pad_w))))
                if m_k.sum() == 0:
                    continue
                for geom, val in rasterio.features.shapes(m_k):
                    if val == 1:
                        s = shapely.geometry.shape(geom)
                        if s.area > 0:
                            pred_shapes.append(s)

        n_patches += 1
        n_pred_per_patch.append(len(pred_shapes))
        n_gt_per_patch.append(len(gt_shapes))
        polygon_deltas.append(abs(len(pred_shapes) - len(gt_shapes)))

        m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)
        for t, (tps, fps, fns) in m["per_t"].items():
            counts[t][0] += tps
            counts[t][1] += fps
            counts[t][2] += fns

        # Reconstruct flattened pred mask for chamfer rasterization.
        pred_bin = np.zeros_like(gt_bin)
        for s in pred_shapes:
            try:
                pred_bin = np.maximum(
                    pred_bin,
                    rasterio.features.rasterize([s], out_shape=gt_bin.shape, dtype=np.uint8),
                )
            except ValueError:
                continue

        for i, j, iou in m["matched_pairs_low"]:
            matched_ious_05.append(iou)
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
        "n_pred_mean": float(np.mean(n_pred_per_patch)) if n_pred_per_patch else 0.0,
        "n_gt_mean": float(np.mean(n_gt_per_patch)) if n_gt_per_patch else 0.0,
        "polygon_count_delta_mean": float(np.mean(polygon_deltas)) if polygon_deltas else 0.0,
        "polygon_count_delta_median": float(np.median(polygon_deltas)) if polygon_deltas else 0.0,
        "boundary_error_m_mean": float(chamfer_m.mean()) if chamfer_m.size else float("nan"),
        "boundary_error_m_p95": (
            float(np.percentile(chamfer_m, 95)) if chamfer_m.size else float("nan")
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--countries", nargs="*", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument(
        "--hf-repo",
        type=str,
        default="torchgeo/delineate-anything",
        help="HuggingFace repo id holding the YOLO11x checkpoint.",
    )
    p.add_argument(
        "--hf-file",
        type=str,
        default="delineate_anything_rgb_yolo11x-88ede029.pt",
    )
    args = p.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"device={device} conf={args.conf} iou={args.iou} imgsz={args.imgsz} split={args.split}")

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

    tag = f"delineate_anything_yolo11x@conf{args.conf}_iou{args.iou}_sz{args.imgsz}"
    countries = args.countries or COUNTRIES
    for country in countries:
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
            f"  pq={m['pq']:.3f} ap={m['ap_5_95']:.3f} "
            f"n_pred/n_gt={m['n_pred_mean']:.1f}/{m['n_gt_mean']:.1f} "
            f"boundary_err_m={m['boundary_error_m_mean']:.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
