"""Appendix figures: bad / okay / great IoU examples per country.

For each country, runs SDF inference on all test patches, computes pixel
field-IoU, and picks 3 patches: worst, median, best. Paginates into
multi-row figures (one country per row -> 3 panels: bad/okay/great with
overlays of prediction).

Writes paper/figs/appendix_bmg_<idx>.pdf, paginated 4 countries per page.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from matplotlib.colors import ListedColormap

from ftw_planet.datasets import FTWPlanet, PLANET_SR_SCALE
from ftw_planet.trainers import SDFSegTask

mpl.rcParams.update({"font.family": "serif", "font.size": 8})

CMAP_SEG = ListedColormap(["#0a3055", "#f0e2bd", "#e68630", "#888888"])

COUNTRIES = [
    "austria","belgium","brazil","cambodia","corsica","croatia","denmark",
    "estonia","finland","france","germany","latvia","lithuania","luxembourg",
    "netherlands","portugal","rwanda","slovakia","slovenia","south_africa",
    "spain","sweden","vietnam",
]


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def field_iou(pred, gt):
    p = pred == 1
    g = gt == 1
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return float("nan")
    return inter / union


@torch.inference_mode()
def predict_idx(task, device, ds, idx, has_sdf):
    s = ds[idx]
    img8 = s["image"]
    gt = s["mask"].numpy()
    x = img8.unsqueeze(0).to(device) / PLANET_SR_SCALE
    x, H, W = _pad32(x, min_size=512)
    if has_sdf:
        seg, _ = task._forward_dual(x)
    else:
        seg = task.model(x)
    pred = seg.softmax(dim=1).argmax(dim=1)[0, :H, :W].cpu().numpy()
    a = img8[4:8].numpy()
    rgb = np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32) / 3000.0
    return ds.records[idx]["patch_id"], np.clip(rgb, 0, 1), gt, pred


def pick_bmg(task, device, country, has_sdf, max_patches=60):
    """Return (worst, median, best) tuples (patch_id, rgb, gt, pred, iou)."""
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    if len(ds) < 3:
        return None
    n = min(len(ds), max_patches)
    idxs = np.linspace(0, len(ds) - 1, n).astype(int).tolist()
    scored = []
    for i in idxs:
        try:
            pid, rgb, gt, pred = predict_idx(task, device, ds, i, has_sdf)
        except Exception as e:
            print(f"    skip {country}[{i}]: {e}")
            continue
        iou = field_iou(pred, gt)
        if np.isnan(iou):
            continue
        # Require min field coverage for fair comparison
        if (gt == 1).mean() < 0.02:
            continue
        scored.append((iou, pid, rgb, gt, pred))
    if len(scored) < 3:
        return None
    scored.sort(key=lambda t: t[0])
    worst = scored[0]
    best = scored[-1]
    median = scored[len(scored) // 2]
    return worst, median, best


def render_page(rows, out_path):
    """rows: list of (country, [worst, median, best])."""
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(11.0, 1.9 * n),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.18})
    if n == 1:
        axes = axes[None, :]
    group_titles = ["Bad (RGB)", "Bad (Pred)", "Okay (RGB)", "Okay (Pred)", "Great (RGB)", "Great (Pred)"]
    for i, (country, triplet) in enumerate(rows):
        for j, (iou, pid, rgb, gt, pred) in enumerate(triplet):
            ax_rgb = axes[i, 2 * j]
            ax_pred = axes[i, 2 * j + 1]
            ax_rgb.imshow(rgb)
            ax_pred.imshow(pred, cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
            ax_rgb.set_title(f"{pid}\nIoU {iou:.2f}", fontsize=7)
            for ax in (ax_rgb, ax_pred):
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
        axes[i, 0].set_ylabel(country, fontsize=9)
        if i == 0:
            for j, t in enumerate(group_titles):
                axes[i, j].set_title(t + "\n" + axes[i, j].get_title(), fontsize=7)
    legend_handles = [
        mpatches.Patch(color="#0a3055", label="bg (0)"),
        mpatches.Patch(color="#f0e2bd", label="field (1)"),
        mpatches.Patch(color="#e68630", label="boundary (2)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({n} countries)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="logs/prue/ftw_planet-unet-efnet3-crop512-sdf/ftw-planet/3e0u1bwd/checkpoints/last.ckpt")
    p.add_argument("--per-page", type=int, default=4)
    p.add_argument("--max-patches", type=int, default=60)
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

    all_rows = []
    for c in COUNTRIES:
        print(f"  {c}...")
        try:
            tr = pick_bmg(task, device, c, has_sdf, max_patches=args.max_patches)
        except Exception as e:
            print(f"    skip {c}: {e}")
            continue
        if tr is None:
            continue
        all_rows.append((c, tr))
        ious = [t[0] for t in tr]
        print(f"    bad/okay/great IoU = {ious[0]:.2f} / {ious[1]:.2f} / {ious[2]:.2f}")

    for i in range(0, len(all_rows), args.per_page):
        chunk = all_rows[i : i + args.per_page]
        idx = i // args.per_page + 1
        render_page(chunk, f"paper/figs/appendix_bmg_{idx}.pdf")


if __name__ == "__main__":
    main()
