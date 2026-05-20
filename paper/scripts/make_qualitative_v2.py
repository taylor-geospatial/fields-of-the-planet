"""Qualitative v2: RGB | RGB + GT overlay | RGB + Pred overlay.

Drops the standalone 3-class mask and watershed-instance columns; the
overlay form is denser per-cell and the eye reads error patterns
directly. Per-row brightness normalization fixes the Rwanda-row
underexposure of v1.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

mpl.rcParams.update({"font.family": "serif", "font.size": 8})

OVERLAY_FIELD = np.array([0.94, 0.86, 0.55])
OVERLAY_BOUND = np.array([0.92, 0.45, 0.10])
ALPHA_FIELD = 0.28
ALPHA_BOUND = 0.85


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def _stretch_per_image(rgb, p_lo=2, p_hi=98):
    """Linear percentile stretch per channel so dark scenes (Rwanda) become readable."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[-1]):
        ch = rgb[..., c].astype(np.float32)
        lo = np.percentile(ch, p_lo)
        hi = np.percentile(ch, p_hi)
        if hi - lo < 1e-6:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def _overlay(rgb, label):
    out = rgb.copy()
    f = label == 1
    b = label == 2
    out[f] = (1.0 - ALPHA_FIELD) * out[f] + ALPHA_FIELD * OVERLAY_FIELD
    out[b] = (1.0 - ALPHA_BOUND) * out[b] + ALPHA_BOUND * OVERLAY_BOUND
    return np.clip(out, 0.0, 1.0)


@torch.inference_mode()
def predict_one(ckpt, country, patch_id, device):
    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    model = task.model.eval().to(device)
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = None
    for i, r in enumerate(ds.records):
        if r["patch_id"] == str(patch_id):
            idx = i
            break
    if idx is None:
        idx = 0
    sample = ds[idx]
    x = sample["image"].unsqueeze(0).to(device)
    y = sample["mask"].cpu().numpy()
    xp, h, w = _pad32(x, min_size=512)
    logits = model(xp)[..., :h, :w]
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Window-A RGB: bands [3,2,1] of the first 4-band stack (B,G,R,NIR -> R,G,B = bands 2,1,0)
    img = sample["image"].cpu().numpy()  # (8, H, W) two windows stacked
    # Channels are [w1_B, w1_G, w1_R, w1_NIR, w2_B, ...]; take w1 RGB.
    rgb = np.stack([img[2], img[1], img[0]], axis=-1) * PLANET_SR_SCALE
    rgb = _stretch_per_image(rgb)
    return country, patch_id, rgb, y.astype(np.uint8), pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="logs/prue/ftw_planet-unet-efnet3-crop512-v3-augmax-full/ftw-planet/mt6mdnl7/checkpoints/last.ckpt",
    )
    p.add_argument(
        "--rows",
        nargs="+",
        default=[
            "austria:g83_00033_11",
            "france:g212_00072_6",
            "rwanda:1592615",
        ],
    )
    p.add_argument("--out", default="paper/figs/qualitative_v2.pdf")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplets = []
    for row in args.rows:
        country, pid = row.split(":", 1)
        triplets.append(predict_one(args.ckpt, country, pid, device))

    n = len(triplets)
    fig, axes = plt.subplots(
        n, 3, figsize=(6.6, 2.25 * n), gridspec_kw={"wspace": 0.04, "hspace": 0.12}
    )
    if n == 1:
        axes = axes[None, :]

    col_titles = ["PlanetScope SR (RGB)", "+ Ground truth", "+ Prediction"]
    for i, (country, pid, rgb, gt, pred) in enumerate(triplets):
        gt_ov = _overlay(rgb, gt)
        pr_ov = _overlay(rgb, pred)
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gt_ov)
        axes[i, 2].imshow(pr_ov)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color("#333333")
        axes[i, 0].set_ylabel(f"{country}\n{pid}", fontsize=7)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
