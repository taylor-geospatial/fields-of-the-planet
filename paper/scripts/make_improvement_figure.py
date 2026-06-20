"""Figure: held-out patches where higher resolution helps most.

Joins the two per-patch metric CSVs (Planet B3, S2 B7) on (country, patch_id),
ranks by ``delta_obj_f1 = planet_obj_f1 - s2_obj_f1``, picks the top patches
with delta>0 and enough GT fields (``--min-n-gt``), and renders one row per
patch:

  S2 RGB (10 m -> Planet grid) | Planet RGB (3 m) | GT instances |
  S2 prediction instances      | Planet prediction instances

Each row is annotated with the country and the two Obj F1 values + delta.

Both models are re-run here (TTA + watershed) using the exact eval functions
so the rendered instances match the CSV metrics. Needs a GPU -> run via
``hpc/per_patch_eval.sbatch``-style sbatch, never the login node.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tg_style
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from postprocess_eval import _pad_min32, _predict_tta, watershed_instances  # noqa: E402

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet  # noqa: E402

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times"],
        "font.size": 8,
        "text.color": tg_style.BROWN,
        "axes.titlecolor": tg_style.BROWN,
        "axes.labelcolor": tg_style.BROWN,
    }
)

S2_NORM_DIVISOR = 3000.0
MASK_BG = np.array(mpl.colors.to_rgb(tg_style.BROWN))


def _stretch(rgb, p_lo=2, p_hi=98):
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[-1]):
        ch = rgb[..., c].astype(np.float32)
        lo, hi = np.percentile(ch, p_lo), np.percentile(ch, p_hi)
        if hi - lo < 1e-6:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def _instance_cmap(n):
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    rng = np.random.default_rng(7)
    colors = np.empty((max(1, n), 3), dtype=np.float32)
    for i in range(max(1, n)):
        c = base[i % 20].copy() + rng.uniform(-0.08, 0.08, size=3)
        colors[i] = np.clip(c, 0, 1)
    return ListedColormap(np.vstack([MASK_BG, colors]))


def _instance_render(inst):
    n = int(inst.max())
    return _instance_cmap(n)(inst)[..., :3]


def _gt_instances(mask):
    field = (mask == 1).astype(np.uint8)
    inst, _ = cc_label(field)
    return inst.astype(np.int32)


def _planet_rgb(country, pid, window):
    p = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(p) as src:
        rgb = src.read([3, 2, 1])  # PSScene BGR(N) -> display R,G,B
    return np.transpose(rgb, (1, 2, 0)).astype(np.float32) / PLANET_SR_SCALE


def _s2_rgb_on_planet_grid(country, pid, window):
    s2 = Path("data/ftw") / country / "s2_images" / f"window_{window}" / f"{pid}.tif"
    planet = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr, h, w = dst.crs, dst.transform, dst.height, dst.width
    with rasterio.open(s2) as src:
        bands = src.read([1, 2, 3])  # FTW S2 is R,G,B,NIR
        out = np.zeros((3, h, w), dtype=bands.dtype)
        for i in range(3):
            reproject(
                source=bands[i],
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_tr,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    return np.transpose(out, (1, 2, 0)).astype(np.float32) / 10000.0


def _predict_instances_planet(task, model, country, pid):
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = next((i for i, r in enumerate(ds.records) if r["patch_id"] == str(pid)), None)
    if idx is None:
        raise RuntimeError(f"planet patch {country}:{pid} not found")
    s = ds[idx]
    image = s["image"].unsqueeze(0).float() / PLANET_SR_SCALE
    mask = s["mask"].unsqueeze(0)
    device = next(model.parameters()).device
    image, mask, H, W = _pad_min32(image.to(device), mask.to(device), min_size=512, pad_mode="zero")
    probs, _sdf = _predict_tta(task, model, image, 20.0)
    seg_np = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    boundary = (seg_np == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    inst = watershed_instances(seg_np, dist, h_min=2.0, field_class=1)
    gt = mask.clone()
    gt[gt == 2] = 0
    gt_np = gt.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    return inst, gt_np


def _predict_instances_s2_on_planet_grid(task, model, country, pid):
    s2_b = Path("data/ftw") / country / "s2_images" / "window_b" / f"{pid}.tif"
    s2_a = Path("data/ftw") / country / "s2_images" / "window_a" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32)
        b_crs, b_tr = src_b.crs, src_b.transform
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32)
    image = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0)).unsqueeze(0) / S2_NORM_DIVISOR
    dummy_mask = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.long)
    device = next(model.parameters()).device
    image, _, H, W = _pad_min32(
        image.to(device), dummy_mask.to(device), min_size=512, pad_mode="zero"
    )
    probs, _sdf = _predict_tta(task, model, image, 20.0)
    seg_np = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    boundary = (seg_np == 2).astype(np.uint8)
    dist = distance_transform_edt(boundary == 0).astype(np.float32)
    inst_s2 = watershed_instances(seg_np, dist, h_min=2.0, field_class=1)
    # Reproject the S2 instance map (native 10m grid) onto the Planet grid (NN).
    planet = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr, dh, dw = dst.crs, dst.transform, dst.height, dst.width
    inst_out = np.zeros((dh, dw), dtype=np.int32)
    reproject(
        source=inst_s2,
        destination=inst_out,
        src_transform=b_tr,
        src_crs=b_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return inst_out


def _load_model(ckpt, device):
    t = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    t = t.eval().to(device)
    return t, t.model


def select_patches(planet_csv, s2_csv, top_n, min_n_gt):
    pl = pd.read_csv(planet_csv)
    s2 = pd.read_csv(s2_csv)
    pl["patch_id"] = pl["patch_id"].astype(str)
    s2["patch_id"] = s2["patch_id"].astype(str)
    j = pl.merge(s2, on=["country", "patch_id"], suffixes=("_pl", "_s2"))
    j["delta_obj_f1"] = j["obj_f1_pl"] - j["obj_f1_s2"]
    # Use the GT count from the Planet side (both share the same label geometry,
    # but Planet GT is at native 3m resolution; n_gt should agree closely).
    sel = j[(j["delta_obj_f1"] > 0) & (j["n_gt_pl"] >= min_n_gt)].copy()
    return sel.sort_values("delta_obj_f1", ascending=False).head(top_n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-csv", default="logs/per_patch/planet_b3.csv")
    p.add_argument("--s2-csv", default="logs/per_patch/s2_b7.csv")
    p.add_argument(
        "--ckpt-planet", default="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    )
    p.add_argument("--ckpt-s2", default="logs/best_checkpoints/s2_efnet7_best.ckpt")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--min-n-gt", type=int, default=8)
    p.add_argument("--window", default="a")
    p.add_argument("--out", default="paper/figs/improvement_examples.pdf")
    p.add_argument("--png", default="logs/improvement_examples.png")
    args = p.parse_args()

    sel = select_patches(args.planet_csv, args.s2_csv, args.top_n, args.min_n_gt)
    if sel.empty:
        raise SystemExit("no patches with delta>0 and enough GT fields; loosen --min-n-gt")
    print("selected patches:")
    for _, r in sel.iterrows():
        print(
            f"  {r['country']}:{r['patch_id']} dF1={r['delta_obj_f1']:.3f} "
            f"(planet={r['obj_f1_pl']:.3f} s2={r['obj_f1_s2']:.3f}) n_gt={int(r['n_gt_pl'])}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading models on {device}...")
    task_pl, model_pl = _load_model(args.ckpt_planet, device)
    task_s2, model_s2 = _load_model(args.ckpt_s2, device)

    rows = []
    for _, r in sel.iterrows():
        country, pid = r["country"], str(r["patch_id"])
        print(f"  rendering {country}:{pid}")
        rgb_s2 = _stretch(_s2_rgb_on_planet_grid(country, pid, args.window))
        rgb_pl = _stretch(_planet_rgb(country, pid, args.window))
        inst_pl, gt_np = _predict_instances_planet(task_pl, model_pl, country, pid)
        inst_s2 = _predict_instances_s2_on_planet_grid(task_s2, model_s2, country, pid)
        gt_inst = _gt_instances(gt_np)
        rows.append(
            {
                "country": country,
                "pid": pid,
                "rgb_s2": rgb_s2,
                "rgb_pl": rgb_pl,
                "gt": gt_inst,
                "inst_s2": inst_s2,
                "inst_pl": inst_pl,
                "f1_pl": r["obj_f1_pl"],
                "f1_s2": r["obj_f1_s2"],
                "delta": r["delta_obj_f1"],
            }
        )

    n = len(rows)
    cols = 5
    col_titles = [
        "Sentinel-2 (10 m)",
        "PlanetScope (3 m)",
        "Ground truth",
        "S2 prediction\nFTW-PRUE (B7)",
        "Planet prediction\nFTP-PRUE",
    ]
    fig, axes = plt.subplots(
        n, cols, figsize=(cols * 1.5, n * 1.62), gridspec_kw={"wspace": 0.02, "hspace": 0.05}
    )
    if n == 1:
        axes = axes[None, :]

    for i, row in enumerate(rows):
        axes[i, 0].imshow(row["rgb_s2"])
        axes[i, 1].imshow(row["rgb_pl"])
        axes[i, 2].imshow(_instance_render(row["gt"]))
        axes[i, 3].imshow(_instance_render(row["inst_s2"]))
        axes[i, 4].imshow(_instance_render(row["inst_pl"]))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color(tg_style.BROWN)
        label = (
            f"{row['country'].replace('_', ' ')}\n"
            f"$\\Delta$F1 = +{row['delta'] * 100:.1f}\n"
            f"(S2 {row['f1_s2'] * 100:.1f} $\\rightarrow$ Planet {row['f1_pl'] * 100:.1f})"
        )
        axes[i, 0].set_ylabel(label, fontsize=6.6, fontweight="bold", linespacing=1.25)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=7.2, fontweight="bold", pad=3, linespacing=1.05)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(args.png, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out} and {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
