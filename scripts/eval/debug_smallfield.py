"""Audit small-field matching against true polygons, per patch, both sensors.

Replicates the exact predict + watershed + match the polygon metric uses, but
for a few patches of one (dense) country, and reports per-patch:
  #true small GT (<0.5 ha), #matched (tp) by Planet, #matched by S2.
Also renders an overlay PNG: Planet RGB | true small polygons | Planet pred
polygons (matched-small green / others grey) | S2 pred polygons. The point is to
SEE whether S2 is legitimately recovering small fields or whether watershed is
tessellating a merged blob into pieces that spuriously match.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polygon_metrics_eval import (
    AP_IOU_THRESHOLDS,
    _eval_grid,
    _extract_shapes,
    _match_shapes,
    _pad_min32,
    _predict_tta,
    _true_gt_shapes,
    watershed_instances,
)

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

SMALL_HA = 0.5
GT_ROOT = "data/ftw_polygons_clipped"


def _load(ckpt, device):
    t = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return t.eval().to(device)


def _watershed_from_seg(seg_np):
    boundary = (seg_np == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    return watershed_instances(seg_np, dist, h_min=2.0, field_class=1)


def _predict_planet(task, ds, idx, device):
    s = ds[idx]
    image = s["image"].unsqueeze(0).float().to(device) / PLANET_SR_SCALE
    mask = s["mask"].unsqueeze(0).to(device)
    image, mask, H, W = _pad_min32(image, mask, min_size=512, pad_mode="zero")
    probs, sdf = _predict_tta(task, task.model, image, 20.0)
    seg = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    return _watershed_from_seg(seg)


def _predict_s2(task, ds, idx, device, up=512):
    s = ds[idx]
    image = s["image"].unsqueeze(0).float().to(device) / 3000.0
    image = F.interpolate(image, size=(up, up), mode="bilinear", align_corners=False)
    mask = torch.zeros((1, up, up), dtype=torch.long, device=device)
    image, mask, H, W = _pad_min32(image, mask, min_size=512, pad_mode="zero")
    probs, sdf = _predict_tta(task, task.model, image, 20.0)
    seg = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    return _watershed_from_seg(seg)


def _small_match(gt_shapes, gt_areas, pred_shapes):
    """Return (n_small_gt, n_small_matched, small_gt_idx, matched_small_gt_idx)."""
    small_idx = [i for i, a in enumerate(gt_areas) if a < SMALL_HA]
    if not gt_shapes or not pred_shapes:
        return len(small_idx), 0, set(small_idx), set()
    m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)
    matched_gt = {i for i, j, _ in m["pairs_per_t"][AP_IOU_THRESHOLDS[0]]}
    matched_small = set(small_idx) & matched_gt
    return len(small_idx), len(matched_small), set(small_idx), matched_small


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="cambodia")
    p.add_argument(
        "--ckpt-planet", default="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    )
    p.add_argument("--ckpt-s2", default="logs/best_checkpoints/s2_efnet7_best.ckpt")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default="logs/debug_smallfield")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

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

    tot = {"gt": 0, "pl": 0, "s2": 0}
    print(f"{'patch':30s} {'#small_gt':>9} {'pl_tp':>6} {'s2_tp':>6}")
    n_done = 0
    for pidx in range(len(pl_ds.records)):
        if n_done >= args.n:
            break
        pid = str(pl_ds.records[pidx]["patch_id"])
        if pid not in s2_by_pid:
            continue
        sidx = s2_by_pid[pid]
        # planet
        inst_pl = _predict_planet(task_pl, pl_ds, pidx, device)
        _, pcrs, ptr, _ = _eval_grid(pl_ds, pidx, args.country, "planet", "data", "a", None)
        gtp, areap = _true_gt_shapes(GT_ROOT, args.country, pid, pcrs, ptr)
        preds_pl = _extract_shapes((inst_pl > 0).astype(np.uint8))
        ng, pl_tp, _, _ = _small_match(gtp, areap, preds_pl)
        # s2
        inst_s2 = _predict_s2(task_s2, s2_ds, sidx, device)
        _, scrs, str_, _ = _eval_grid(s2_ds, sidx, args.country, "s2", "data", "a", 512)
        gts, areas = _true_gt_shapes(GT_ROOT, args.country, pid, scrs, str_)
        preds_s2 = _extract_shapes((inst_s2 > 0).astype(np.uint8))
        ngs, s2_tp, _, _ = _small_match(gts, areas, preds_s2)
        print(f"{pid:30s} {ng:9d} {pl_tp:6d} {s2_tp:6d}   (s2 #small_gt={ngs})")
        tot["gt"] += ng
        tot["pl"] += pl_tp
        tot["s2"] += s2_tp
        n_done += 1
    print(
        f"\nTOTAL {args.country}: small_gt(planet-grid)={tot['gt']} "
        f"planet_tp={tot['pl']} ({tot['pl'] / max(tot['gt'], 1):.1%}) "
        f"s2_tp={tot['s2']} ({tot['s2'] / max(tot['gt'], 1):.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
