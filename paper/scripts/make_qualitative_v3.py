"""Qualitative v3: more rows, side-by-side Planet vs S2 predictions.

Columns per row:
  1. PlanetScope RGB
  2. Sentinel-2 RGB (FTW chip, reprojected onto the Planet patch grid)
  3. Ground truth (3-class overlay on Planet RGB)
  4. Our prediction  (Planet B3 augmax full, WS+TTA - on Planet RGB)
  5. S2 prediction   (S2 B7 augmax full, WS+TTA  - on Planet RGB)

Rows are picked to span the storytelling space:
  * Austria          (clean, large fields)
  * France smallholder mosaic (typical)
  * Rwanda fragmented fields (hard transfer)
  * Cambodia tiny parcels (where S2 wins; useful to show the failure mode)
  * Lithuania       (Nordic, S2 wins on pixel-IoU but loses on instances)
  * Denmark / Sweden (showcase Planet win on clean fields)

Per-row brightness stretch normalizes contrast across countries.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from rasterio.warp import Resampling, reproject

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial"],
    "font.size": 8,
})

OVERLAY_FIELD = np.array([0.94, 0.86, 0.55])
OVERLAY_BOUND = np.array([0.92, 0.45, 0.10])
ALPHA_FIELD = 0.28
ALPHA_BOUND = 0.85

S2_NORM_DIVISOR = 3000.0


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def _stretch(rgb, p_lo=2, p_hi=98):
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


def _load_planet_sample(country, pid):
    """Return (planet_rgb_stretched, gt_mask, input_tensor_8ch, native_HW)."""
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = None
    for i, r in enumerate(ds.records):
        if r["patch_id"] == str(pid):
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"patch {country}:{pid} not found")
    sample = ds[idx]
    x = sample["image"]  # numpy or tensor (8, H, W)
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    y = sample["mask"]
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    y = np.asarray(y).astype(np.uint8)
    rgb_t = torch.stack([x[2], x[1], x[0]], dim=-1).float()
    rgb = rgb_t.numpy() * PLANET_SR_SCALE
    rgb = _stretch(rgb)
    return rgb, y, x.float(), sample


def _load_s2_rgb_aligned_to_planet(ftw_root: Path, planet_root: Path,
                                   country: str, pid: str, window: str):
    """Read FTW S2 chip and reproject to the matching Planet patch's grid for
    a visually aligned RGB thumbnail."""
    s2 = ftw_root / country / "s2_images" / f"window_{window}" / f"{pid}.tif"
    sr = planet_root / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(sr) as dst:
        dst_crs, dst_transform = dst.crs, dst.transform
        h, w = dst.height, dst.width
    with rasterio.open(s2) as src:
        bands = src.read([1, 2, 3])  # R, G, B
        out = np.zeros((3, h, w), dtype=bands.dtype)
        for i in range(3):
            reproject(
                source=bands[i],
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    rgb = np.transpose(out, (1, 2, 0)).astype(np.float32) / S2_NORM_DIVISOR
    return _stretch(np.clip(rgb, 0, 1))


@torch.inference_mode()
def _predict(model, x, device):
    xp, h, w = _pad32(x.unsqueeze(0).to(device), min_size=512)
    logits = model(xp)[..., :h, :w]
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _load_task(ckpt, device):
    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return task.model.eval().to(device)


def _predict_s2(model_s2, country: str, pid: str, device):
    """Run the S2 model on the FTW S2 sample for this patch, reproject the
    prediction onto the Planet grid so it can be overlaid on the Planet RGB."""
    from ftw_tools.training.datamodules import FTWDataModule  # local import (heavy)
    # Simpler path: load the FTW S2 sample directly via rasterio + the same
    # 8-channel stacking (window B then A) that the S2 trainer expects.
    ftw_root = Path("data/ftw")
    s2_a = ftw_root / country / "s2_images" / "window_a" / f"{pid}.tif"
    s2_b = ftw_root / country / "s2_images" / "window_b" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32) / 10000.0
        b_crs, b_tr, b_h, b_w = src_b.crs, src_b.transform, src_b.height, src_b.width
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32) / 10000.0
    # Stack window B then A (matches FTW trainer convention).
    x = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0))
    pred_s2 = _predict(model_s2, x, device)  # on S2 grid (10 m)
    # Reproject pred to Planet grid.
    planet_sr = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet_sr) as dst:
        dst_crs, dst_tr, dst_h, dst_w = dst.crs, dst.transform, dst.height, dst.width
    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=pred_s2,
        destination=out,
        src_transform=b_tr,
        src_crs=b_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt-planet",
        default="logs/prue/ftw_planet-unet-efnet3-crop512-v3-augmax-full/ftw-planet/mt6mdnl7/checkpoints/last.ckpt",
    )
    p.add_argument(
        "--ckpt-s2",
        default="logs/prue/ftw_s2-unet-efnet7-crop256-s2-v3-augmax-b7-full/ftw-s2/2x26jpwu/checkpoints/last.ckpt",
    )
    p.add_argument(
        "--rows",
        nargs="+",
        default=[
            "austria:g83_00033_11:a",
            "france:g68_00021_4:a",
            "denmark:g6_00086_10:b",
            "lithuania:g25_00016_7:b",
            "rwanda:1592589:a",
            "cambodia:g172_00010_3:a",
        ],
    )
    p.add_argument("--out", default="paper/figs/qualitative_v3.pdf")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading models on {device}...")
    model_pl = _load_task(args.ckpt_planet, device)
    model_s2 = _load_task(args.ckpt_s2, device)

    ftw_root = Path("data/ftw")
    planet_root = Path("data/planet")

    rows = []
    for spec in args.rows:
        parts = spec.split(":")
        country, pid = parts[0], parts[1]
        window = parts[2] if len(parts) > 2 else "a"
        print(f"  {country}:{pid} (window {window})")
        rgb, gt, x, sample = _load_planet_sample(country, pid)
        s2_rgb = _load_s2_rgb_aligned_to_planet(ftw_root, planet_root, country, pid, window)
        pred_pl = _predict(model_pl, x, device)
        pred_s2 = _predict_s2(model_s2, country, pid, device)
        rows.append((country, pid, rgb, s2_rgb, gt, pred_pl, pred_s2))

    n = len(rows)
    cols = 5
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.65, n * 1.7),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.08})
    if n == 1:
        axes = axes[None, :]

    col_titles = [
        "Planet RGB (3 m)",
        "S2 RGB (10 m)",
        "Ground truth",
        "Ours (Planet)",
        "Baseline (S2)",
    ]
    for i, (country, pid, rgb, s2_rgb, gt, pred_pl, pred_s2) in enumerate(rows):
        gt_ov   = _overlay(rgb,    gt)
        pl_ov   = _overlay(rgb,    pred_pl)
        s2_pred = _overlay(rgb,    pred_s2)
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(s2_rgb)
        axes[i, 2].imshow(gt_ov)
        axes[i, 3].imshow(pl_ov)
        axes[i, 4].imshow(s2_pred)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color("#444444")
        axes[i, 0].set_ylabel(country.replace("_", " "),
                              fontsize=8.5, fontweight="bold")
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8.5, fontweight="bold", pad=3)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
