"""Qualitative v6: square-crop, same-window S2/Planet, viridis field overlay.

For each row:
  1. Pick a (country, patch_id, window) where Planet beats S2 on smallholder
     fields and the patch has high field-class coverage.
  2. Load PlanetScope RGB for that window.
  3. Reproject the FTW S2 chip for the SAME window onto the Planet UTM grid.
     (Both seasons stay aligned: window_a = plant, window_b = harvest --
      whichever we pick, both modalities show the same season.)
  4. Center-crop everything to a square, resample to a common pixel size.
  5. Overlay the field-interior class (class 1) as a plasma colormap blend
     on the RGB.  No boundary line clutter -- the colored field interior
     carries the visual message.
  6. Predict with FTP-PRUE (Planet) and FTW-PRUE (B7) (S2 baseline).

Layout: 5 columns x N rows
  Planet RGB | S2 RGB (aligned) | Ground truth | FTP-PRUE (ours) | FTW-PRUE (B7)

All overlays sit on the Planet RGB so the same scene anchors the comparison.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tg_style
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from rasterio.warp import Resampling, reproject

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

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

S2_NORM_DIVISOR = 3000.0  # model input normalization; not a display knob
S2_UPSAMPLE = 512  # bilinear-upsample S2 256->512 (corrected resize_factor=2 protocol)
FIELD_GREEN = np.array(mpl.colors.to_rgb(tg_style.GREEN))  # brand green for field
MASK_BG = np.array(mpl.colors.to_rgb(tg_style.BROWN))  # brand brown for bg + boundary
SQUARE_SIZE = 256  # final pixel size for every cell


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


def _square_crop(arr):
    """Center-crop along the longer axis so output is square."""
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side]


def _resize_nn(arr, size):
    """Resize to (size, size).  uint8 masks use NEAREST (preserve class id);
    float arrays (RGB, probability heatmaps) use BILINEAR."""
    from PIL import Image

    if arr.dtype == np.uint8 and arr.ndim == 2:
        return np.array(Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST))
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    out = np.array(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR)) / 255.0
    return out.astype(np.float32)


def _hard_mask_overlay(mask):
    """Render a 3-class mask as a hard segmentation image: green for field
    interior (class 1), black for background + boundary (classes 0 and 2).
    Returns float RGB in [0, 1]."""
    out = np.empty((*mask.shape, 3), dtype=np.float32)
    out[:] = MASK_BG
    out[mask == 1] = FIELD_GREEN
    return out


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


@torch.inference_mode()
def _predict(model, x, device, scale):
    """Return per-pixel argmax class id (uint8) in {0,1,2}."""
    x = x.float() / scale
    xp, h, w = _pad32(x.unsqueeze(0).to(device), min_size=512)
    logits = model(xp)[..., :h, :w]
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _load_task(ckpt, device):
    t = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return t.model.eval().to(device)


def _load_planet(country, pid):
    """Return (rgb_stretched_full, gt_mask_full, x_8ch_full).

    x_8ch_full = [w_b(4), w_a(4)] tensor on the patch grid."""
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = next((i for i, r in enumerate(ds.records) if r["patch_id"] == str(pid)), None)
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
    return x.float(), y


def _planet_rgb_for_window(country, pid, window):
    """Load the requested Planet window directly from disk so we don't depend
    on the dataset's stacking order. Returns float RGB on the patch grid."""
    p = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(p) as src:
        bgr_nir = src.read([3, 2, 1])  # R, G, B
        return np.transpose(bgr_nir, (1, 2, 0)).astype(np.float32) / PLANET_SR_SCALE


def _s2_rgb_for_window(country, pid, window):
    """Reproject the FTW S2 chip for `window_{window}` onto the Planet patch
    grid. Returns float RGB normalized by 10000 (FTW S2 native DN scale)."""
    s2 = Path("data/ftw") / country / "s2_images" / f"window_{window}" / f"{pid}.tif"
    planet = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_transform = dst.crs, dst.transform
        h, w = dst.height, dst.width
    with rasterio.open(s2) as src:
        bands = src.read([1, 2, 3])
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
    return np.transpose(out, (1, 2, 0)).astype(np.float32) / 10000.0


def _predict_s2_to_planet_grid(model_s2, country, pid, device):
    """Run the S2 baseline (B+A stacked) and reproject the prediction back
    to the Planet grid for visual alignment.

    Uses the corrected resize_factor=2 protocol: bilinear-upsample the stacked
    256 input to 512 before inference (matches the headline tables), then map
    the 512 prediction onto the Planet grid via a transform scaled by 256/512.
    """
    s2_a = Path("data/ftw") / country / "s2_images" / "window_a" / f"{pid}.tif"
    s2_b = Path("data/ftw") / country / "s2_images" / "window_b" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32)
        b_crs, b_tr, bh, bw = src_b.crs, src_b.transform, src_b.height, src_b.width
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32)
    x = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0)).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(S2_UPSAMPLE, S2_UPSAMPLE), mode="bilinear", align_corners=False)
    pred_s2 = _predict(model_s2, x.squeeze(0), device, scale=S2_NORM_DIVISOR)
    # The 512 prediction maps onto the native 256 S2 grid via a transform
    # scaled by 256/512; build it from the source 256 transform.
    up_tr = b_tr * rasterio.Affine.scale(bw / S2_UPSAMPLE, bh / S2_UPSAMPLE)
    planet = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr, dst_h, dst_w = dst.crs, dst.transform, dst.height, dst.width
    out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=pred_s2,
        destination=out,
        src_transform=up_tr,
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
        default="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt",
    )
    p.add_argument(
        "--ckpt-s2",
        default="logs/best_checkpoints/s2_efnet7_best.ckpt",
    )
    # 12 dense-label, smallholder-leaning patches; window picked to match the
    # season we want shown.  All are countries where macro Planet > S2.
    p.add_argument(
        "--rows",
        nargs="+",
        default=[
            # 7 dense-smallholder rows so the figure + caption fit one page.
            "croatia:g10-3_00071_11:a",
            "croatia:g10-3_00015_7:a",
            "slovenia:g13_00033_1:a",
            "austria:g83_00031_18:a",
            "austria:g83_00019_5:a",
            "lithuania:g11_00088_0:a",
            "finland:g15-1_00141_12:a",
        ],
    )
    p.add_argument("--out", default="paper/figs/qualitative_v6_appx.pdf")
    p.add_argument("--cell-size", type=int, default=SQUARE_SIZE)
    p.add_argument(
        "--cell-h",
        type=float,
        default=1.38,
        help="Inches per row.  Reduce for compact banner figures.",
    )
    p.add_argument("--cell-w", type=float, default=1.35, help="Inches per column.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading models on {device}...")
    model_pl = _load_task(args.ckpt_planet, device)
    model_s2 = _load_task(args.ckpt_s2, device)

    rows = []
    for spec in args.rows:
        parts = spec.split(":")
        country, pid = parts[0], parts[1]
        window = parts[2] if len(parts) > 2 else "a"
        print(f"  {country}:{pid} window={window}")
        # Planet input + GT mask (full grid).
        x8, gt_full = _load_planet(country, pid)
        # Planet RGB for the chosen window (may differ from the dataset's
        # window-B-first stacking convention; we read straight from disk).
        rgb_pl = _planet_rgb_for_window(country, pid, window)
        rgb_s2 = _s2_rgb_for_window(country, pid, window)
        # Predict argmax class id (uint8).
        pred_pl = _predict(model_pl, x8, device, scale=PLANET_SR_SCALE)
        pred_s2 = _predict_s2_to_planet_grid(model_s2, country, pid, device)
        # Square-crop everything to the same centered square, then resize.
        sq_size = args.cell_size
        rgb_pl_s = _resize_nn(_stretch(_square_crop(rgb_pl)), sq_size)
        rgb_s2_s = _resize_nn(_stretch(_square_crop(rgb_s2)), sq_size)
        gt_s = _resize_nn(_square_crop(gt_full).astype(np.uint8), sq_size)
        pred_pl_s = _resize_nn(_square_crop(pred_pl), sq_size)
        pred_s2_s = _resize_nn(_square_crop(pred_s2), sq_size)
        rows.append((country, pid, window, rgb_pl_s, rgb_s2_s, gt_s, pred_pl_s, pred_s2_s))

    n = len(rows)
    cols = 5
    _fig, axes = plt.subplots(
        n,
        cols,
        figsize=(cols * args.cell_w, n * args.cell_h),
        gridspec_kw={"wspace": 0.015, "hspace": 0.03},
    )
    if n == 1:
        axes = axes[None, :]
    col_titles = [
        "Planet RGB (3 m)",
        "S2 RGB (10 m)",
        "Ground truth",
        "FTP-PRUE+",
        "FTW-PRUE+",
    ]
    for i, (country, pid, window, rgb_pl, rgb_s2, gt, pred_pl, pred_s2) in enumerate(rows):
        axes[i, 0].imshow(rgb_pl)
        axes[i, 1].imshow(rgb_s2)
        axes[i, 2].imshow(_hard_mask_overlay(gt))
        axes[i, 3].imshow(_hard_mask_overlay(pred_pl))
        axes[i, 4].imshow(_hard_mask_overlay(pred_s2))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color(tg_style.BROWN)
        axes[i, 0].set_ylabel(country.replace("_", " "), fontsize=7.5, fontweight="bold")
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8.5, fontweight="bold", pad=3)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
