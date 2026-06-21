"""Diagnostic figure: why PlanetScope under-segments fields in Germany.

Germany is the one held-out region where Planet loses to the S2 baseline.
Prior analysis showed Planet gets field *area* right (pixel IoU ties S2) but
**merges adjacent parcels** (n_pred/n_gt ~ 0.49; RQ/SQ/boundary all worse),
and the deficit is worst on patches with many parcels. This figure lets us
*see* that: it ranks Germany patches with >=10 GT fields by the most negative
``delta_obj_f1 = planet_obj_f1 - s2_obj_f1`` and renders one row per patch:

  PlanetScope RGB (3 m) | Sentinel-2 RGB (10 m) | GT instances |
  Planet prediction      | S2 prediction

Each row is annotated with country, GT field count, and the two Obj F1 values.

Both models are re-run (TTA + watershed) with the *exact* protocols behind the
per-patch CSVs so the rendered instances match the numbers:

* Planet: native 512, FTWPlanet backend, /PLANET_SR_SCALE   (planet_b3.csv).
* S2: bilinear-upsample 256->512 (resize_factor=2, mask nearest), /3000, then
  pad — the corrected eval protocol  (s2_b7_upsampled512.csv). NOT zero-pad.

Watershed uses the predicted SDF head when present, else EDT of the boundary
class, with h_min=2.0 / sdf_clip=20.0 — same as scripts/eval/per_patch_metrics.

Needs a GPU -> run via hpc/germany_diagnostic.sbatch, never the login node.
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

from postprocess_eval import (  # noqa: E402
    _has_sdf_head,
    _pad_min32,
    _predict_tta,
    watershed_instances,
)

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
S2_UPSAMPLE = 512
MIN_PAD = 512
SDF_CLIP = 20.0
H_MIN = 2.0
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


def _square_crop(arr):
    """Center-crop along the longer axis so output is square."""
    h, w = arr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side]


def _resize_square(arr, size):
    """Square-crop then resize to (size, size). Integer label maps use NEAREST
    (preserve instance ids); float RGB uses bilinear."""
    from PIL import Image

    arr = _square_crop(arr)
    if np.issubdtype(arr.dtype, np.integer):
        im = Image.fromarray(arr.astype(np.int32), mode="I")
        return np.array(im.resize((size, size), Image.Resampling.NEAREST)).astype(arr.dtype)
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    out = np.array(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR)) / 255.0
    return out.astype(np.float32)


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


def _watershed_from_probs(probs, sdf, H, W):
    seg_np = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    if sdf is not None:
        dist = sdf.squeeze(0).cpu().numpy().astype(np.float32)[:H, :W]
    else:
        boundary = (seg_np == 2).astype(np.uint8)
        dist = distance_transform_edt(boundary == 0).astype(np.float32)
    return watershed_instances(seg_np, dist, h_min=H_MIN, field_class=1)


def _predict_instances_planet(task, model, country, pid):
    """Planet @ native 512 — mirrors planet_b3.csv (FTWPlanet backend)."""
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = next((i for i, r in enumerate(ds.records) if r["patch_id"] == str(pid)), None)
    if idx is None:
        raise RuntimeError(f"planet patch {country}:{pid} not found")
    s = ds[idx]
    image = s["image"].unsqueeze(0).float() / PLANET_SR_SCALE
    mask = s["mask"].unsqueeze(0)
    device = next(model.parameters()).device
    image, mask, H, W = _pad_min32(
        image.to(device), mask.to(device), min_size=MIN_PAD, pad_mode="zero"
    )
    probs, sdf = _predict_tta(task, model, image, SDF_CLIP)
    inst = _watershed_from_probs(probs, sdf, H, W)
    gt = mask.clone()
    gt[gt == 2] = 0
    gt_np = gt.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
    return inst, gt_np


def _predict_instances_s2_on_planet_grid(task, model, country, pid):
    """S2 @ upsample-512 — mirrors s2_b7_upsampled512.csv.

    bilinear-upsample stacked (window_b, window_a) 256->512, /3000, pad, predict,
    watershed, then reproject the 512 instance map back onto the Planet grid.
    """
    s2_b = Path("data/ftw") / country / "s2_images" / "window_b" / f"{pid}.tif"
    s2_a = Path("data/ftw") / country / "s2_images" / "window_a" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32)
        b_crs, b_tr, bh, bw = src_b.crs, src_b.transform, src_b.height, src_b.width
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32)
    image = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0)).unsqueeze(0) / S2_NORM_DIVISOR
    device = next(model.parameters()).device
    image = image.to(device)
    image = torch.nn.functional.interpolate(
        image, size=(S2_UPSAMPLE, S2_UPSAMPLE), mode="bilinear", align_corners=False
    )
    dummy = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=torch.long, device=device)
    image, _, H, W = _pad_min32(image, dummy, min_size=MIN_PAD, pad_mode="zero")
    probs, sdf = _predict_tta(task, model, image, SDF_CLIP)
    inst512 = _watershed_from_probs(probs, sdf, H, W)
    # The 512 instance map maps onto the native 256 S2 grid via a transform
    # scaled by 256/512; build it from the source 256 transform.
    sx, sy = bw / S2_UPSAMPLE, bh / S2_UPSAMPLE
    up_tr = b_tr * rasterio.Affine.scale(sx, sy)
    planet = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr, dh, dw = dst.crs, dst.transform, dst.height, dst.width
    inst_out = np.zeros((dh, dw), dtype=np.int32)
    reproject(
        source=inst512,
        destination=inst_out,
        src_transform=up_tr,
        src_crs=b_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return inst_out


def _load_model(ckpt, device):
    t = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    t = t.eval().to(device)
    print(f"  loaded {Path(ckpt).name} (sdf_head={_has_sdf_head(t)})")
    return t, t.model


def select_patches(planet_csv, s2_csv, top_n, min_n_gt, country):
    pl = pd.read_csv(planet_csv)
    s2 = pd.read_csv(s2_csv)
    pl["patch_id"] = pl["patch_id"].astype(str)
    s2["patch_id"] = s2["patch_id"].astype(str)
    j = pl.merge(s2, on=["country", "patch_id"], suffixes=("_pl", "_s2"))
    j["delta_obj_f1"] = j["obj_f1_pl"] - j["obj_f1_s2"]
    sel = j[(j["country"] == country) & (j["n_gt_pl"] >= min_n_gt)].copy()
    # Most negative delta first = worst Planet-vs-S2 patches.
    return sel.sort_values("delta_obj_f1", ascending=True).head(top_n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-csv", default="logs/per_patch/planet_b3.csv")
    p.add_argument("--s2-csv", default="logs/per_patch/s2_b7_upsampled512.csv")
    p.add_argument(
        "--ckpt-planet", default="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    )
    p.add_argument("--ckpt-s2", default="logs/best_checkpoints/s2_efnet7_best.ckpt")
    p.add_argument("--country", default="germany")
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--min-n-gt", type=int, default=10)
    p.add_argument("--sq-size", type=int, default=512, help="Square-crop+resize each panel to this.")
    p.add_argument("--window", default="a")
    p.add_argument("--out", default="paper/figs/germany_diagnostic.pdf")
    p.add_argument("--png", default="logs/germany_diagnostic.png")
    args = p.parse_args()

    sel = select_patches(args.planet_csv, args.s2_csv, args.top_n, args.min_n_gt, args.country)
    if sel.empty:
        raise SystemExit(f"no {args.country} patches with n_gt>={args.min_n_gt}; loosen --min-n-gt")
    print("selected worst patches (most negative planet-s2 obj_f1):")
    for _, r in sel.iterrows():
        print(
            f"  {r['country']}:{r['patch_id']} dF1={r['delta_obj_f1']:.3f} "
            f"(planet={r['obj_f1_pl']:.3f} s2={r['obj_f1_s2']:.3f}) "
            f"n_gt={int(r['n_gt_pl'])} n_pred_pl={int(r['n_pred_pl'])} n_pred_s2={int(r['n_pred_s2'])}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading models on {device}...")
    task_pl, model_pl = _load_model(args.ckpt_planet, device)
    task_s2, model_s2 = _load_model(args.ckpt_s2, device)

    sq = args.sq_size
    rows = []
    for _, r in sel.iterrows():
        country, pid = r["country"], str(r["patch_id"])
        print(f"  rendering {country}:{pid}")
        rgb_pl = _stretch(_planet_rgb(country, pid, args.window))
        rgb_s2 = _stretch(_s2_rgb_on_planet_grid(country, pid, args.window))
        inst_pl, gt_np = _predict_instances_planet(task_pl, model_pl, country, pid)
        inst_s2 = _predict_instances_s2_on_planet_grid(task_s2, model_s2, country, pid)
        gt_inst = _gt_instances(gt_np)
        rows.append(
            {
                "country": country,
                "pid": pid,
                "rgb_pl": _resize_square(rgb_pl, sq),
                "rgb_s2": _resize_square(rgb_s2, sq),
                "gt": _resize_square(gt_inst, sq),
                "inst_pl": _resize_square(inst_pl, sq),
                "inst_s2": _resize_square(inst_s2, sq),
                "n_gt": int(r["n_gt_pl"]),
                "f1_pl": r["obj_f1_pl"],
                "f1_s2": r["obj_f1_s2"],
            }
        )

    n = len(rows)
    cols = 5
    col_titles = [
        "PlanetScope (3 m)",
        "Sentinel-2 (10 m)",
        "Ground truth",
        "Planet prediction\nFTP-PRUE",
        "S2 prediction\nFTW-PRUE (B7)",
    ]
    fig, axes = plt.subplots(
        n, cols, figsize=(cols * 1.5, n * 1.62), gridspec_kw={"wspace": 0.02, "hspace": 0.05}
    )
    if n == 1:
        axes = axes[None, :]

    for i, row in enumerate(rows):
        axes[i, 0].imshow(row["rgb_pl"])
        axes[i, 1].imshow(row["rgb_s2"])
        axes[i, 2].imshow(_instance_render(row["gt"]))
        axes[i, 3].imshow(_instance_render(row["inst_pl"]))
        axes[i, 4].imshow(_instance_render(row["inst_s2"]))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color(tg_style.BROWN)
        label = (
            f"{row['country'].replace('_', ' ')}\n"
            f"{row['n_gt']} fields\n"
            f"Planet {row['f1_pl'] * 100:.1f} vs S2 {row['f1_s2'] * 100:.1f}"
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
