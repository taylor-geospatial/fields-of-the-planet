"""Worked-example figure for the polygon metrics (Fig. metric_example).

Renders a single PlanetScope patch as [image | GT field instances | predicted
field instances] and prints the per-patch polygon metrics next to it, so the
reader can see what object F1 / PQ / SQ / boundary chamfer actually measure.

Inference (Planet PRUE-FTP-B3 augmax) and the metric computation reuse the
exact functions from ``scripts/eval`` so the example numbers match the paper's
evaluation protocol. Runs on CPU; one patch is a few seconds.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tg_style
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from polygon_metrics_eval import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    GSD_M,
    _boundary_pixels,
    _extract_shapes,
    _match_shapes,
    _symmetric_chamfer,
)
from postprocess_eval import _pad_min32, _predict_tta, gt_instances, watershed_instances  # noqa: E402

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times"]})
NORM_DIVISOR = 3000.0


def _f1(tps: int, fps: int, fns: int) -> float:
    p = tps / max(tps + fps, 1)
    r = tps / max(tps + fns, 1)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def patch_metrics(seg_np: np.ndarray, gt_np: np.ndarray, dist: np.ndarray, gsd_m: float) -> dict:
    """Per-patch PQ/SQ/RQ/AP/chamfer/|dN|/pixel-IoU on the field class."""
    inst_pred = watershed_instances(seg_np, dist, h_min=2.0, field_class=1)
    pred_bin = (inst_pred > 0).astype(np.uint8)
    gt_bin = (gt_np == 1).astype(np.uint8)

    gt_shapes = _extract_shapes(gt_bin)
    pred_shapes = _extract_shapes(pred_bin)
    m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)

    t05 = AP_IOU_THRESHOLDS[0]
    tps, fps, fns = m["per_t"][t05]
    rq = _f1(tps, fps, fns)
    matched_ious = [iou for _, _, iou in m["matched_pairs_low"]]
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    pq = sq * rq
    ap = float(np.mean([_f1(*m["per_t"][t]) for t in AP_IOU_THRESHOLDS]))

    chamfers = []
    import rasterio.features

    for i, j, _ in m["matched_pairs_low"]:
        pb = _boundary_pixels(
            rasterio.features.rasterize([pred_shapes[j]], out_shape=pred_bin.shape, dtype=np.uint8)
        )
        gb = _boundary_pixels(
            rasterio.features.rasterize([gt_shapes[i]], out_shape=gt_bin.shape, dtype=np.uint8)
        )
        c = _symmetric_chamfer(pb, gb)
        if c is not None:
            chamfers.append(c)
    bnd_m = float(np.mean(chamfers)) * gsd_m if chamfers else float("nan")

    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    pix_iou = inter / union if union else 0.0

    return {
        "n_gt": len(gt_shapes),
        "n_pred": len(pred_shapes),
        "objf1": rq,
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "ap": ap,
        "bnd_m": bnd_m,
        "dN": abs(len(pred_shapes) - len(gt_shapes)),
        "pix_iou": pix_iou,
        "inst_pred": inst_pred,
    }


def _inst_cmap(n: int, seed: int = 0) -> mpl.colors.ListedColormap:
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    rng = np.random.default_rng(seed)
    cols = base[rng.integers(0, 20, size=max(n, 1))]
    cols = np.clip(cols + rng.uniform(-0.08, 0.08, cols.shape), 0, 1)
    return mpl.colors.ListedColormap(np.vstack([[1, 1, 1], cols]))


def show_instances(ax, inst: np.ndarray, title: str) -> None:
    n = int(inst.max())
    remap = np.zeros(inst.max() + 1, dtype=np.int32)
    remap[1:] = np.arange(1, n + 1)
    ax.imshow(remap[inst], cmap=_inst_cmap(n, seed=n), interpolation="nearest")
    ax.set_title(title, fontsize=10, color=tg_style.BROWN, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=REPO / "logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt")
    ap.add_argument("--root", type=str, default=str(REPO / "data"))
    ap.add_argument("--country", type=str, default="croatia")
    ap.add_argument("--max-patches", type=int, default=80)
    ap.add_argument("--n-show", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "paper/figs/metric_example.pdf")
    args = ap.parse_args()

    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    task = task.eval()
    model = task.model
    gsd_m = GSD_M["planet"]

    ds = FTWPlanet(
        root=args.root, countries=[args.country], split="test", transforms=None, load_boundaries=True
    )

    cands = []
    for idx in range(min(args.max_patches, len(ds))):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).float() / PLANET_SR_SCALE
        mask = sample["mask"].unsqueeze(0)
        image, mask, H, W = _pad_min32(image, mask, min_size=512, pad_mode="zero")
        probs, _ = _predict_tta(task, model, image, 20.0)
        seg_np = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        boundary_np = (seg_np == 2).astype(np.uint8)
        dist = distance_transform_edt(boundary_np == 0).astype(np.float32)
        met = patch_metrics(seg_np, gt_np, dist, gsd_m)

        # RGB from window-B channels [B,G,R,NIR] -> display [R,G,B].
        rgb = image[0, [2, 1, 0]].cpu().numpy()[:, :H, :W] * PLANET_SR_SCALE
        rgb = np.clip(np.transpose(rgb, (1, 2, 0)) / NORM_DIVISOR, 0, 1)

        # Need enough fields to be interesting and a finite boundary error.
        if not (8 <= met["n_gt"] <= 30) or not np.isfinite(met["bnd_m"]):
            continue
        cands.append({"idx": idx, "rgb": rgb, "gt": gt_np, "met": met})
        print(f"  cand idx={idx} n_gt={met['n_gt']} objf1={met['objf1']:.3f} pq={met['pq']:.3f}")

    n_show = args.n_show
    if len(cands) < n_show:
        raise SystemExit(f"only {len(cands)} candidates in first {args.max_patches} of {args.country}")

    # Pick patches spanning the recognition range (a strong, a middling, and a
    # harder case) so the figure shows how the metrics move with quality.
    cands.sort(key=lambda c: c["met"]["objf1"])
    quantiles = np.linspace(0.85, 0.2, n_show)  # high -> low obj F1
    picks = [cands[int(round(q * (len(cands) - 1)))] for q in quantiles]

    fig, axes = plt.subplots(
        n_show, 4, figsize=(10.5, 2.85 * n_show), gridspec_kw={"width_ratios": [1, 1, 1, 0.95]}
    )
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for r, pick in enumerate(picks):
        met = pick["met"]
        gt_inst = gt_instances(pick["gt"], field_class=1)
        axes[r, 0].imshow(pick["rgb"])
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        show_instances(axes[r, 1], gt_inst, "Ground-truth fields" if r == 0 else "")
        show_instances(axes[r, 2], met["inst_pred"], "Predicted fields" if r == 0 else "")
        if r == 0:
            axes[r, 0].set_title("PlanetScope (3 m)", fontsize=10, color=tg_style.BROWN, pad=3)

        axes[r, 3].axis("off")
        rows = [
            ("Object F1 @ 0.5 IoU", f"{met['objf1']:.3f}"),
            ("PQ  (= SQ x RQ)", f"{met['pq']:.3f}"),
            ("   SQ (mean matched IoU)", f"{met['sq']:.3f}"),
            ("   RQ (recognition)", f"{met['rq']:.3f}"),
            ("F1 [.5:.95]", f"{met['ap']:.3f}"),
            ("Boundary chamfer", f"{met['bnd_m']:.1f} m"),
            ("|N_pred - N_gt|", f"{met['dN']}  ({met['n_pred']} vs {met['n_gt']})"),
            ("Pixel IoU (field)", f"{met['pix_iou']:.3f}"),
        ]
        y = 0.97
        for name, val in rows:
            axes[r, 3].text(0.0, y, name, fontsize=8.5, color=tg_style.BROWN, transform=axes[r, 3].transAxes)
            axes[r, 3].text(1.0, y, val, fontsize=8.5, ha="right", color=tg_style.BROWN, transform=axes[r, 3].transAxes)
            y -= 0.118

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}  (country={args.country}, idxs={[p['idx'] for p in picks]})")
    for p in picks:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p["met"].items() if k != "inst_pred"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
