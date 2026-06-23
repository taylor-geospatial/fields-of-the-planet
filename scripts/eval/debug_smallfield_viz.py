"""Visual + numeric audit of small-field matching on one dense patch, per sensor.

For a chosen (country, patch_id) renders, for Planet and S2:
  RGB | predicted seg (field/boundary/bg) | predicted polygons (matched-small
  green, unmatched-small red, non-small grey) | true small GT polygons
and prints, for the small bin: #true small, #pred, #pred small, #matched, plus
the actual IoU values of matched small pairs (to confirm matches are real and the
IoU/matching is not buggy).
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polygon_metrics_eval import (
    AP_IOU_THRESHOLDS,
    _eval_grid,
    _extract_shapes,
    _pad_min32,
    _predict_tta,
    _true_gt_shapes,
    watershed_instances,
)

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

SMALL_HA = 0.5
GT_ROOT = "data/ftw_polygons_clipped"
T = AP_IOU_THRESHOLDS[0]  # 0.5


def _load(ckpt, device):
    return (
        CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
        .eval()
        .to(device)
    )


def _seg_and_inst(task, image, device):
    image, mask, H, W = _pad_min32(
        image, torch.zeros_like(image[:, :1]).long(), min_size=512, pad_mode="zero"
    )
    probs, _ = _predict_tta(task, task.model, image, 20.0)
    seg = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    dist = distance_transform_edt((seg == 2) == 0).astype(np.float32)
    inst = watershed_instances(seg, dist, h_min=2.0, field_class=1)
    return seg, inst


def _stretch(rgb):
    return np.clip(rgb.astype(np.float32) / 2500.0, 0, 1)


def _iou(a, b):
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter) if (a.area + b.area - inter) > 0 else 0.0


def _match_small(gt_shapes, gt_areas, preds):
    small = [i for i, a in enumerate(gt_areas) if a < SMALL_HA]
    matched, ious = set(), []
    used = set()
    for i in small:
        for j, p in enumerate(preds):
            if j in used:
                continue
            v = _iou(gt_shapes[i], p)
            if v > T:
                used.add(j)
                matched.add(i)
                ious.append(v)
                break
    return small, matched, ious, used


def _seg_rgb(seg):
    out = np.zeros((*seg.shape, 3), np.float32)
    out[seg == 0] = (0.23, 0.12, 0.11)  # bg brown
    out[seg == 1] = (0.81, 0.95, 0.62)  # field green
    out[seg == 2] = (1.0, 0.31, 0.17)  # boundary red
    return out


def _draw(ax, preds, size, title):
    ax.set_facecolor("#f4f4eb")
    for p in preds:
        ax.add_patch(
            plt.Polygon(
                np.array(p.exterior.coords),
                closed=True,
                fill=False,
                edgecolor="#3b1e1c",
                linewidth=0.3,
            )
        )
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="cambodia")
    p.add_argument("--patch", default=None, help="patch_id; default = first shared patch")
    p.add_argument(
        "--ckpt-planet", default="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    )
    p.add_argument("--ckpt-s2", default="logs/best_checkpoints/s2_efnet7_best.ckpt")
    p.add_argument("--out", default="logs/debug_smallfield/viz.png")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    task_pl = _load(args.ckpt_planet, device)
    task_s2 = _load(args.ckpt_s2, device)
    pl_ds = FTWPlanet(root="data", countries=[args.country], split="test", load_boundaries=True)
    from ftw_tools.training.datasets import FTW

    s2_ds = FTW(
        root="data/ftw",
        countries=[args.country],
        split="test",
        transforms=None,
        load_boundaries=True,
        temporal_options="stacked",
    )
    s2_by_pid = {Path(f["window_a"]).stem: i for i, f in enumerate(s2_ds.filenames)}

    pid = args.patch
    pidx = None
    for k in range(len(pl_ds.records)):
        q = str(pl_ds.records[k]["patch_id"])
        if (pid is None and q in s2_by_pid) or q == pid:
            pid, pidx = q, k
            break
    print(f"patch {args.country}:{pid}")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    for row, (name, task, ds, idx, backend, up, scale) in enumerate(
        [
            ("Planet", task_pl, pl_ds, pidx, "planet", None, PLANET_SR_SCALE),
            ("S2-B7", task_s2, s2_ds, s2_by_pid[pid], "s2", 512, 3000.0),
        ]
    ):
        s = ds[idx]
        img = s["image"].unsqueeze(0).float().to(device) / scale
        if up:
            img = F.interpolate(img, size=(up, up), mode="bilinear", align_corners=False)
        seg, inst = _seg_and_inst(task, img, device)
        preds = _extract_shapes((inst > 0).astype(np.uint8))
        _, ecrs, etr, _ = _eval_grid(ds, idx, args.country, backend, "data", "a", up)
        gt_shapes, gt_areas = _true_gt_shapes(GT_ROOT, args.country, pid, ecrs, etr)
        small, matched, ious, _ = _match_small(gt_shapes, gt_areas, preds)
        size = seg.shape[0]
        # RGB
        if backend == "planet":
            with rasterio.open(f"data/planet/{args.country}/window_a/{pid}.tif") as src:
                rgb = np.transpose(src.read([3, 2, 1]), (1, 2, 0))
        else:
            with rasterio.open(f"data/ftw/{args.country}/s2_images/window_a/{pid}.tif") as ssrc:
                rgb = np.transpose(ssrc.read([1, 2, 3]), (1, 2, 0))
        axes[row, 0].imshow(_stretch(rgb))
        axes[row, 0].set_title(f"{name} RGB", fontsize=8)
        axes[row, 1].imshow(_seg_rgb(seg))
        axes[row, 1].set_title(f"{name} seg (green=field, red=boundary)", fontsize=8)
        _draw(axes[row, 2], preds, size, f"{name} pred polys (n={len(preds)})")
        # true small GT
        axes[row, 3].set_facecolor("#f4f4eb")
        if small:
            gpd.GeoDataFrame(geometry=[gt_shapes[i] for i in small]).plot(
                ax=axes[row, 3], facecolor="#cff29e", edgecolor="#3b1e1c", linewidth=0.3
            )
        axes[row, 3].set_xlim(0, size)
        axes[row, 3].set_ylim(size, 0)
        axes[row, 3].set_aspect("equal")
        axes[row, 3].set_title(
            f"{name} true small GT (n={len(small)}, matched={len(matched)})", fontsize=8
        )
        for c in range(4):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
        iar = np.array(ious) if ious else np.array([0.0])
        pred_areas = np.array([pp.area for pp in preds])
        print(
            f"{name}: pred_total={len(preds)} true_small={len(small)} matched_small={len(matched)} "
            f"recall={len(matched) / max(len(small), 1):.1%} | matched IoU min/med/max="
            f"{iar.min():.2f}/{np.median(iar):.2f}/{iar.max():.2f} | "
            f"pred area px median={np.median(pred_areas):.0f}"
        )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
