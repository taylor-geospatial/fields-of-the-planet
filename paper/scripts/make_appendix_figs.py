"""Render appendix qualitative figures: one row per country, paginated.

For each country we pick a representative test patch (mid-size, ~50%
field coverage when available, otherwise just the largest non-empty
patch) and render: RGB / GT / Prediction / Watershed instances.

Writes paper/figs/appendix_qual_<idx>.pdf, with 5 countries per figure.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet
from ftw_planet.trainers import SDFSegTask

mpl.rcParams.update({"font.family": "serif", "font.size": 8})

CMAP_SEG = ListedColormap(["#0a3055", "#f0e2bd", "#e68630", "#888888"])

COUNTRIES = [
    "austria",
    "belgium",
    "brazil",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "portugal",
    "rwanda",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def pick_patch(ds, target_frac=0.4, min_side=400) -> int:
    """Best-effort: a mid-size patch with non-trivial field coverage."""
    best = (-1.0, 0)
    for i, r in enumerate(ds.records):
        with rasterio.open(r["label"]) as src:
            m = src.read(1)
        h, w = m.shape
        if h < min_side or w < min_side // 2:
            continue
        f = (m == 1).mean()
        if f < 0.05:
            continue
        score = -abs(f - target_frac)
        if score > best[0]:
            best = (score, i)
    return best[1]


@torch.inference_mode()
def predict(task, model, device, country, has_sdf):
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    if len(ds) == 0:
        return None
    idx = pick_patch(ds)
    s = ds[idx]
    img8 = s["image"]
    gt = s["mask"].numpy()
    x = img8.unsqueeze(0).to(device) / PLANET_SR_SCALE
    x, H, W = _pad32(x, min_size=512)
    if has_sdf:
        seg, sdf = task._forward_dual(x)
        sdf_np = sdf[0, :H, :W].cpu().numpy() * task.sdf_clip
    else:
        seg = model(x)
        sdf_np = None
    pred = seg.softmax(dim=1).argmax(dim=1)[0, :H, :W].cpu().numpy()
    # watershed
    if sdf_np is not None:
        dist = sdf_np
    else:
        boundary = (pred == 2).astype(np.uint8)
        dist = distance_transform_edt(boundary == 0).astype(np.float32)
    field_mask = pred == 1
    seeds = h_maxima(dist * field_mask, h=2.0)
    markers, _ = label(seeds)
    if markers.max() == 0:
        markers, _ = label(field_mask)
    inst = watershed(-dist, markers=markers, mask=field_mask)
    a = img8[4:8].numpy()
    rgb = np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32) / 3000.0
    return ds.records[idx]["patch_id"], np.clip(rgb, 0, 1), gt, pred, inst


def render_figure(rows, out_path):
    n = len(rows)
    fig, axes = plt.subplots(
        n, 4, figsize=(7.0, 2.0 * n), gridspec_kw={"wspace": 0.05, "hspace": 0.15}
    )
    if n == 1:
        axes = axes[None, :]
    col_titles = [
        "PlanetScope SR (RGB)",
        "Ground truth",
        "Prediction (3-class)",
        "Watershed instances",
    ]
    for i, (country, pid, rgb, gt, pred, inst) in enumerate(rows):
        rng = np.random.default_rng(0)
        n_inst = int(inst.max())
        tab20 = plt.get_cmap("tab20")
        ic = [tab20(i / 20) for i in range(20)] * max(1, n_inst // 20 + 1)
        ic = rng.permutation(np.array(ic))[: max(n_inst, 1)]
        inst_cmap = ListedColormap([[0, 0, 0]] + list(ic))
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gt, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 2].imshow(pred, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 3].imshow(inst, cmap=inst_cmap, interpolation="nearest")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
        axes[i, 0].set_ylabel(f"{country}\n{pid}", fontsize=7)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8)
    legend_handles = [
        mpatches.Patch(color="#0a3055", label="bg (0)"),
        mpatches.Patch(color="#f0e2bd", label="field interior (1)"),
        mpatches.Patch(color="#e68630", label="boundary (2)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({n} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default="logs/prue/ftw_planet-unet-efnet3-crop512-sdf/ftw-planet/3e0u1bwd/checkpoints/last.ckpt",
    )
    p.add_argument("--per-figure", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        task = SDFSegTask.load_from_checkpoint(args.ckpt, map_location="cpu")
        has_sdf = True
        print("loaded SDFSegTask")
    except Exception:
        task = CustomSemanticSegmentationTask.load_from_checkpoint(args.ckpt, map_location="cpu")
        has_sdf = False
        print("loaded base task")
    task = task.eval().to(device)
    model = task.model

    rows = []
    for c in COUNTRIES:
        try:
            r = predict(task, model, device, c, has_sdf)
        except Exception as e:
            print(f"  skip {c}: {e}")
            continue
        if r is None:
            continue
        pid, rgb, gt, pred, inst = r
        rows.append((c, pid, rgb, gt, pred, inst))
        print(f"  {c}: {pid}")

    # Paginate
    for i in range(0, len(rows), args.per_figure):
        chunk = rows[i : i + args.per_figure]
        idx = i // args.per_figure + 1
        render_figure(chunk, f"paper/figs/appendix_qual_{idx}.pdf")


if __name__ == "__main__":
    main()
