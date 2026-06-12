"""S2 PRUE-B7 vs FTP SDF comparison on identical patches.

For each FTW country (intersection of S2 + Planet test sets):
  1. Compute per-country pixel field IoU for both models.
  2. Pick one representative patch (largest with mid field coverage) and
     render: S2 RGB | Planet RGB | GT | S2 Pred | Planet Pred.

Writes:
  paper/figs/s2_vs_planet_table.tex     LaTeX table
  paper/figs/s2_vs_planet_iou.csv       per-country numbers
  paper/figs/s2_vs_planet_qual_<i>.pdf  qualitative rows, paginated
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from matplotlib.colors import ListedColormap

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet
from ftw_planet.trainers import SDFSegTask

mpl.rcParams.update({"font.family": "serif", "font.size": 8})

CMAP_SEG = ListedColormap(["#0a3055", "#f0e2bd", "#e68630", "#888888"])
S2_SCALE = 3000.0

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


def field_iou(pred, gt):
    p = pred == 1
    g = gt == 1
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


def load_s2(country, patch_id):
    """Return (img8 uint16 tensor [8,H,W], mask3 [H,W], rgb [H,W,3])."""
    root = Path("data/ftw") / country
    wa = root / "s2_images/window_a" / f"{patch_id}.tif"
    wb = root / "s2_images/window_b" / f"{patch_id}.tif"
    m = root / "label_masks/semantic_3class" / f"{patch_id}.tif"
    if not (wa.exists() and wb.exists() and m.exists()):
        return None
    with rasterio.open(wa) as src:
        a = src.read()  # (4,H,W)
    with rasterio.open(wb) as src:
        b = src.read()
    with rasterio.open(m) as src:
        mk = src.read(1)
    # ftw uses window B first, then A
    img8 = np.concatenate([b, a], axis=0).astype(np.float32)
    rgb_a = np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32) / S2_SCALE
    return img8, mk, np.clip(rgb_a, 0, 1)


@torch.inference_mode()
def predict_s2(s2_model, device, img8):
    """S2 PRUE evaluation: single forward pass at native 256x256 patch size
    (matching the FTW baseline eval.py protocol)."""
    x = torch.from_numpy(img8).unsqueeze(0).to(device) / S2_SCALE
    H, W = x.shape[-2:]
    # only pad to next mult-32 if patch isn't already
    if H % 32 or W % 32:
        x, _, _ = _pad32(x, min_size=0)
    seg = s2_model(x).softmax(dim=1)
    return seg.argmax(dim=1)[0, :H, :W].cpu().numpy()


@torch.inference_mode()
def predict_planet(task, device, sample):
    img8 = sample["image"]
    x = img8.unsqueeze(0).to(device) / PLANET_SR_SCALE
    x, H, W = _pad32(x, min_size=512)
    seg, _ = task._forward_dual(x)
    pred = seg.softmax(dim=1).argmax(dim=1)[0, :H, :W].cpu().numpy()
    a = img8[4:8].numpy()
    rgb = np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32) / 3000.0
    return pred, np.clip(rgb, 0, 1)


def run_country(country, s2_model, planet_task, device, max_patches=60):
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    if len(ds) == 0:
        return None
    idxs = np.linspace(0, len(ds) - 1, min(len(ds), max_patches)).astype(int).tolist()
    rows = []
    for i in idxs:
        rec = ds.records[i]
        pid = rec["patch_id"]
        s2 = load_s2(country, pid)
        if s2 is None:
            continue
        img8_s2, mk_s2, rgb_s2 = s2
        try:
            ps = ds[i]
        except Exception:
            continue
        gt_pl = ps["mask"].numpy()
        # Each model scored against its OWN co-registered GT (S2 and Planet
        # labels are rasterized on slightly different grids -- cross-grid
        # pixel scoring is invalid).
        pred_s2 = predict_s2(s2_model, device, img8_s2)
        pred_pl_full, rgb_pl = predict_planet(planet_task, device, ps)
        Hs, Ws = mk_s2.shape
        Hp, Wp = gt_pl.shape
        pred_s2 = pred_s2[:Hs, :Ws]
        pred_pl = pred_pl_full[:Hp, :Wp]
        if (mk_s2 == 1).sum() < 100 and (gt_pl == 1).sum() < 100:
            continue
        iou_s2 = field_iou(pred_s2, mk_s2)
        iou_pl = field_iou(pred_pl, gt_pl)
        rows.append(
            {
                "patch_id": pid,
                "iou_s2": iou_s2,
                "iou_planet": iou_pl,
                "rgb_s2": rgb_s2[:Hs, :Ws],
                "rgb_pl": rgb_pl[:Hp, :Wp],
                "gt_s2": mk_s2,
                "gt_pl": gt_pl,
                "pred_s2": pred_s2,
                "pred_pl": pred_pl,
                "ff": float((mk_s2 == 1).mean()),
                "size": Hs * Ws,
            }
        )
    if not rows:
        return None
    iou_s2 = np.nanmean([r["iou_s2"] for r in rows])
    iou_pl = np.nanmean([r["iou_planet"] for r in rows])
    # pick a representative patch: largest size, mid field coverage
    scored = sorted(rows, key=lambda r: (-r["size"], abs(r["ff"] - 0.4)))
    return iou_s2, iou_pl, len(rows), scored[0]


def render_qual(rows, out_path):
    n = len(rows)
    fig, axes = plt.subplots(
        n, 5, figsize=(10.5, 2.1 * n), gridspec_kw={"wspace": 0.04, "hspace": 0.18}
    )
    if n == 1:
        axes = axes[None, :]
    col_titles = ["S2 RGB", "Planet RGB", "GT (Planet)", "S2 PRUE-B7 pred", "FTP SDF pred"]
    for i, (country, sample) in enumerate(rows):
        axes[i, 0].imshow(sample["rgb_s2"])
        axes[i, 1].imshow(sample["rgb_pl"])
        axes[i, 2].imshow(sample["gt_pl"], cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 3].imshow(sample["pred_s2"], cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 4].imshow(sample["pred_pl"], cmap=CMAP_SEG, vmin=0, vmax=3, interpolation="nearest")
        axes[i, 3].set_title(f"IoU {sample['iou_s2']:.2f}", fontsize=7)
        axes[i, 4].set_title(f"IoU {sample['iou_planet']:.2f}", fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
        axes[i, 0].set_ylabel(f"{country}\n{sample['patch_id']}", fontsize=7)
        if i == 0:
            for j, t in enumerate(col_titles):
                cur = axes[i, j].get_title()
                axes[i, j].set_title((t + ("\n" + cur if cur else "")), fontsize=8)
    legend_handles = [
        mpatches.Patch(color="#0a3055", label="bg (0)"),
        mpatches.Patch(color="#f0e2bd", label="field (1)"),
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
        "--s2-ckpt", default="/u/isaaccorley/.cache/torch/hub/checkpoints/FTW_PRUE_EFNET_B7.ckpt"
    )
    p.add_argument(
        "--planet-ckpt",
        default="logs/prue/ftw_planet-unet-efnet3-crop512-sdf/ftw-planet/3e0u1bwd/checkpoints/last.ckpt",
    )
    p.add_argument("--per-page", type=int, default=6)
    p.add_argument("--max-patches", type=int, default=60)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading S2 PRUE-B7...")
    s2_task = CustomSemanticSegmentationTask.load_from_checkpoint(args.s2_ckpt, map_location="cpu")
    s2_model = s2_task.model.eval().to(device)
    print("loading Planet SDF...")
    pl_task = (
        SDFSegTask.load_from_checkpoint(args.planet_ckpt, map_location="cpu").eval().to(device)
    )

    table_rows = []
    qual_rows = []
    for c in COUNTRIES:
        print(f"  {c}...")
        try:
            out = run_country(c, s2_model, pl_task, device, max_patches=args.max_patches)
        except Exception as e:
            print(f"    skip {c}: {e}")
            continue
        if out is None:
            continue
        iou_s2, iou_pl, n, rep = out
        table_rows.append({"country": c, "n": n, "iou_s2": iou_s2, "iou_planet": iou_pl})
        qual_rows.append((c, rep))
        print(f"    n={n}  S2 IoU={iou_s2:.3f}  Planet IoU={iou_pl:.3f}  ({rep['patch_id']})")

    df = pd.DataFrame(table_rows)
    df.to_csv("paper/figs/s2_vs_planet_iou.csv", index=False)
    print(f"wrote paper/figs/s2_vs_planet_iou.csv ({len(df)} rows)")
    print(f"mean S2 IoU: {df.iou_s2.mean():.3f}   mean Planet IoU: {df.iou_planet.mean():.3f}")

    # LaTeX table
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Country & N & S2 PRUE-B7 & FTP SDF \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        better = r"\textbf"
        s2 = f"{r['iou_s2']:.3f}"
        pl = f"{r['iou_planet']:.3f}"
        if r["iou_planet"] > r["iou_s2"]:
            pl = f"{better}{{{pl}}}"
        elif r["iou_s2"] > r["iou_planet"]:
            s2 = f"{better}{{{s2}}}"
        lines.append(f"{r['country'].replace('_', ' ')} & {int(r['n'])} & {s2} & {pl} \\\\")
    lines += [
        r"\midrule",
        f"Mean & -- & {df.iou_s2.mean():.3f} & \\textbf{{{df.iou_planet.mean():.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    Path("paper/figs/s2_vs_planet_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote paper/figs/s2_vs_planet_table.tex")

    for i in range(0, len(qual_rows), args.per_page):
        chunk = qual_rows[i : i + args.per_page]
        idx = i // args.per_page + 1
        render_qual(chunk, f"paper/figs/s2_vs_planet_qual_{idx}.pdf")


if __name__ == "__main__":
    main()
