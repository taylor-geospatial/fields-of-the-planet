"""Render a paper-quality qualitative figure: RGB / GT / Pred / Watershed.

Picks one representative patch per row. Use --country / --patch-id to
override defaults. Saves paper/figs/qualitative.pdf.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from ftw_planet.datasets import FTWPlanet, PLANET_SR_SCALE

mpl.rcParams.update({"font.family": "serif", "font.size": 8})

# 0 = bg (dark blue), 1 = field (light cream), 2 = boundary (orange)
CMAP_SEG = ListedColormap(["#0a3055", "#f0e2bd", "#e68630", "#888888"])


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


@torch.inference_mode()
def predict_one(ckpt, country, patch_id, device):
    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    model = task.model.eval().to(device)
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = None
    for i, r in enumerate(ds.records):
        if r["patch_id"] == str(patch_id):
            idx = i; break
    if idx is None:
        idx = 0
        patch_id = ds.records[0]["patch_id"]
    s = ds[idx]
    img8 = s["image"]
    gt = s["mask"].numpy()
    x = img8.unsqueeze(0).to(device) / PLANET_SR_SCALE
    x, H, W = _pad32(x, min_size=512)
    seg = model(x).softmax(dim=1)
    pred = seg.argmax(dim=1)[0, :H, :W].cpu().numpy()
    # watershed via distance_transform on the predicted boundary
    boundary = (pred == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    field_mask = pred == 1
    seeds = h_maxima(dist * field_mask, h=2.0)
    markers, _ = label(seeds)
    if markers.max() == 0:
        markers, _ = label(field_mask)
    inst = watershed(-dist, markers=markers, mask=field_mask)
    # RGB
    a = img8[4:8].numpy()  # window A bands: B, G, R, NIR
    rgb = np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32) / 3000.0
    rgb = np.clip(rgb, 0, 1)
    return country, patch_id, rgb, gt, pred, inst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="logs/prue/ftw_planet-unet-efnet3-bf16/ftw-planet/1io8addz/checkpoints/last.ckpt")
    p.add_argument("--rows", nargs="+", default=[
        "austria:g83_00033_11",
        "france:g68_00021_4",
        "rwanda:1592589",
    ])
    p.add_argument("--out", default="paper/figs/qualitative.pdf")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    triplets = []
    for row in args.rows:
        country, pid = row.split(":", 1)
        triplets.append(predict_one(args.ckpt, country, pid, device))

    n = len(triplets)
    fig, axes = plt.subplots(n, 4, figsize=(7.0, 2.0 * n), gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    if n == 1: axes = axes[None, :]
    col_titles = ["PlanetScope SR (RGB)", "Ground truth", "Prediction (3-class)", "Watershed instances"]
    for i, (country, pid, rgb, gt, pred, inst) in enumerate(triplets):
        # Use a perturbed colormap for instance ids so adjacent labels are distinct.
        rng = np.random.default_rng(0)
        n_inst = int(inst.max())
        tab20 = plt.get_cmap("tab20")
        tab20_colors = [tab20(i / 20) for i in range(20)]
        ic = rng.permutation(tab20_colors * max(1, n_inst // 20 + 1))[: max(n_inst, 1)]
        inst_cmap = ListedColormap([[0, 0, 0]] + list(ic))

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gt, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 2].imshow(pred, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 3].imshow(inst, cmap=inst_cmap, interpolation="nearest")
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_linewidth(0.4)
        axes[i, 0].set_ylabel(f"{country}\n{pid}", fontsize=7)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8)

    # legend below
    legend_handles = [
        mpatches.Patch(color="#0a3055", label="bg (0)"),
        mpatches.Patch(color="#f0e2bd", label="field interior (1)"),
        mpatches.Patch(color="#e68630", label="boundary (2)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
