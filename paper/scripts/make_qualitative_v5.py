"""Qualitative v5: plasma-heatmap overlay on the field-interior class.

Layout (per row): Planet RGB | Sentinel-2 RGB | GT plasma overlay |
PRUE-HD-B3 plasma overlay | PRUE-B7 (S2) plasma overlay.

The overlay encodes class 1 (field interior) as a plasma colormap blended
on the RGB at alpha 0.55, with class 2 (boundary) drawn as a thin red
contour.  Bg pixels show the raw RGB.  Diagnostic: every prediction
array is saved to /tmp/qual_diag_*.npy so we can inspect whether
predictions actually cover the patch or only the top/bottom strips.
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
from scipy.ndimage import binary_dilation

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times"],
    "font.size": 8,
})

S2_NORM_DIVISOR = 3000.0
PLASMA = plt.get_cmap("plasma")


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
        lo, hi = np.percentile(ch, p_lo), np.percentile(ch, p_hi)
        if hi - lo < 1e-6:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def _plasma_overlay(rgb, mask, alpha=0.55, boundary_color=(0.93, 0.20, 0.10)):
    """Blend a plasma color onto pixels where mask==1 (field interior).
    Draw a 1-px contour for mask==2 (boundary class)."""
    out = rgb.copy()
    field = (mask == 1).astype(np.float32)
    # Plasma at value 0.55 is a vivid magenta-orange; we use a fixed
    # color rather than a per-pixel intensity (the prediction is binary).
    field_color = np.array(PLASMA(0.55)[:3])
    fmask = field.astype(bool)
    out[fmask] = (1 - alpha) * out[fmask] + alpha * field_color
    # Boundary as thin contour
    bmask = mask == 2
    edge = bmask & ~binary_dilation(bmask, iterations=0)  # always full bmask = thin already
    # Lay it slightly thicker for visibility
    edge = binary_dilation(bmask, iterations=0) | bmask
    bcolor = np.array(boundary_color)
    out[bmask] = bcolor
    return np.clip(out, 0, 1)


@torch.inference_mode()
def _predict(model, x, device, scale=PLANET_SR_SCALE):
    """Normalize by `scale` to match training preprocessing
    (training uses std=PLANET_SR_SCALE for Planet, 3000 for S2)."""
    x = x.float() / scale
    xp, h, w = _pad32(x.unsqueeze(0).to(device), min_size=512)
    logits = model(xp)[..., :h, :w]
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _load_task(ckpt, device):
    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return task.model.eval().to(device)


def _load_planet_sample(country, pid):
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = None
    for i, r in enumerate(ds.records):
        if r["patch_id"] == str(pid):
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"patch {country}:{pid} not found")
    s = ds[idx]
    x = s["image"]
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    y = s["mask"]
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    y = np.asarray(y).astype(np.uint8)
    rgb_t = torch.stack([x[2], x[1], x[0]], dim=-1).float()
    rgb = _stretch(rgb_t.numpy() * PLANET_SR_SCALE)
    return rgb, y, x.float()


def _load_s2_rgb(ftw_root, planet_root, country, pid, window):
    s2 = ftw_root / country / "s2_images" / f"window_{window}" / f"{pid}.tif"
    sr = planet_root / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(sr) as dst:
        dst_crs, dst_transform = dst.crs, dst.transform
        h, w = dst.height, dst.width
    with rasterio.open(s2) as src:
        bands = src.read([1, 2, 3])
        out = np.zeros((3, h, w), dtype=bands.dtype)
        for i in range(3):
            reproject(source=bands[i], destination=out[i],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
    rgb = np.transpose(out, (1, 2, 0)).astype(np.float32) / S2_NORM_DIVISOR
    return _stretch(np.clip(rgb, 0, 1))


def _predict_s2(model_s2, country, pid, device):
    ftw_root = Path("data/ftw")
    s2_a = ftw_root / country / "s2_images" / "window_a" / f"{pid}.tif"
    s2_b = ftw_root / country / "s2_images" / "window_b" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32)
        b_crs, b_tr, b_h, b_w = src_b.crs, src_b.transform, src_b.height, src_b.width
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32)
    x = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0))
    # FTW S2 norm divisor = 3000 (the FTW PRUE convention).
    pred_s2 = _predict(model_s2, x, device, scale=3000.0)
    planet_sr = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet_sr) as dst:
        dst_crs, dst_tr, dst_h, dst_w = dst.crs, dst.transform, dst.height, dst.width
    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(source=pred_s2, destination=out,
              src_transform=b_tr, src_crs=b_crs,
              dst_transform=dst_tr, dst_crs=dst_crs,
              resampling=Resampling.nearest)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-planet",
                   default="logs/prue/ftw_planet-unet-efnet3-crop512-v3-augmax-full/ftw-planet/mt6mdnl7/checkpoints/last.ckpt")
    p.add_argument("--ckpt-s2",
                   default="logs/prue/ftw_s2-unet-efnet7-crop256-s2-v3-augmax-b7-full/ftw-s2/2x26jpwu/checkpoints/last.ckpt")
    p.add_argument("--rows", nargs="+",
                   default=["south_africa:g1_00010_17:a", "lithuania:g11_00037_4:a",
                            "sweden:g6-0_00031_11:a", "denmark:g22_00013_4:a",
                            "latvia:g25_00013_14:a", "lithuania:g11_00040_9:b"])
    p.add_argument("--out", default="paper/figs/qualitative_v5.pdf")
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
        print(f"  {country}:{pid} ({window})")
        rgb, gt, x = _load_planet_sample(country, pid)
        s2_rgb = _load_s2_rgb(ftw_root, planet_root, country, pid, window)
        pred_pl = _predict(model_pl, x, device)
        pred_s2 = _predict_s2(model_s2, country, pid, device)
        # Diagnostic dump
        np.save(f"/tmp/qual_diag_{country}_{pid}_planet.npy", pred_pl)
        np.save(f"/tmp/qual_diag_{country}_{pid}_s2.npy", pred_s2)
        np.save(f"/tmp/qual_diag_{country}_{pid}_gt.npy", gt)
        print(f"    shape rgb={rgb.shape} gt={gt.shape} pred_pl={pred_pl.shape} pred_s2={pred_s2.shape}")
        print(f"    gt class counts {np.bincount(gt.ravel(), minlength=4)} "
              f"pred_pl {np.bincount(pred_pl.ravel(), minlength=4)} "
              f"pred_s2 {np.bincount(pred_s2.ravel(), minlength=4)}")
        rows.append((country, pid, rgb, s2_rgb, gt, pred_pl, pred_s2))

    n = len(rows)
    cols = 5
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.7, n * 1.75),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Planet RGB (3 m)", "S2 RGB (10 m)", "Ground truth",
                  "PRUE-HD-B3 (ours)", "PRUE-B7 (S2 baseline)"]
    for i, (country, pid, rgb, s2_rgb, gt, pred_pl, pred_s2) in enumerate(rows):
        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(s2_rgb)
        axes[i, 2].imshow(_plasma_overlay(rgb, gt))
        axes[i, 3].imshow(_plasma_overlay(rgb, pred_pl))
        axes[i, 4].imshow(_plasma_overlay(rgb, pred_s2))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color("#444444")
        axes[i, 0].set_ylabel(country.replace("_", " "), fontsize=8.5, fontweight="bold")
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8.5, fontweight="bold", pad=3)
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
