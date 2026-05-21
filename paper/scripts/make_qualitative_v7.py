"""Qualitative v7: post-processed (TTA + Watershed) instance segmentations.

Same dense-smallholder rows as v6, but the predicted-mask columns are run
through the full inference stack used by our headline numbers:

  D4 8-way test-time augmentation (TTA) -> argmax  ->  marker-controlled
  watershed seeded by h-maxima of the field-class distance transform.

The output is an *instance* label image, not a class mask.  We render each
connected component as a distinct color (perturbed tab20).  Background is
black.

Layout (5 columns x N rows):
  Planet RGB | S2 RGB | GT instances | PRUE-HD-B3 (TTA+WS) | PRUE-B7 S2 (TTA+WS)
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
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times"],
        "font.size": 8,
    }
)

S2_NORM_DIVISOR = 3000.0
SQUARE_SIZE = 256


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
    h, w = arr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side]


def _resize_nn(arr, size):
    from PIL import Image

    if arr.dtype == np.uint8 and arr.ndim == 2:
        return np.array(Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST))
    if arr.ndim == 2 and arr.dtype.kind in "iu":
        return np.array(
            Image.fromarray(arr.astype(np.int32)).resize((size, size), Image.Resampling.NEAREST)
        )
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    out = np.array(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR)) / 255.0
    return out.astype(np.float32)


def _instance_cmap(n):
    """Build a categorical colormap with n+1 entries (idx 0 = black bg)."""
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    rng = np.random.default_rng(7)
    colors = np.empty((max(1, n), 3), dtype=np.float32)
    for i in range(max(1, n)):
        c = base[i % 20].copy()
        c = c + rng.uniform(-0.08, 0.08, size=3)
        colors[i] = np.clip(c, 0, 1)
    return ListedColormap(np.vstack([[0, 0, 0], colors]))


def _instance_render(inst):
    n = int(inst.max())
    cmap = _instance_cmap(n)
    return cmap(inst)[..., :3]


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def _d4_transforms():
    """Eight D4 (forward, inverse) pairs for TTA on (B, C, H, W) tensors."""

    def _flip(x, dims):
        return torch.flip(x, dims=dims) if dims else x

    def _rot(k):
        return (
            lambda x: torch.rot90(x, k, dims=(-2, -1)),
            lambda x: torch.rot90(x, -k, dims=(-2, -1)),
        )

    yield (lambda x: x, lambda x: x)
    yield _rot(1)
    yield _rot(2)
    yield _rot(3)
    yield (lambda x: _flip(x, [-1]), lambda x: _flip(x, [-1]))
    yield (lambda x: _flip(x, [-2]), lambda x: _flip(x, [-2]))
    yield (lambda x: _flip(x, [-2, -1]), lambda x: _flip(x, [-2, -1]))
    yield (
        lambda x: torch.rot90(_flip(x, [-1]), 1, dims=(-2, -1)),
        lambda x: _flip(torch.rot90(x, -1, dims=(-2, -1)), [-1]),
    )


@torch.inference_mode()
def _predict_probs_tta(model, x, device, scale, tta=True):
    """Return averaged softmax probabilities (C, H, W) after D4 TTA."""
    x = (x.float() / scale).unsqueeze(0).to(device)
    xp, h, w = _pad32(x, min_size=512)
    if not tta:
        return torch.softmax(model(xp), dim=1).squeeze(0)[..., :h, :w].cpu().numpy()
    probs_sum = None
    n = 0
    for fwd, inv in _d4_transforms():
        out = model(fwd(xp))
        p = torch.softmax(out, dim=1)
        p = inv(p)
        probs_sum = p if probs_sum is None else probs_sum + p
        n += 1
    assert probs_sum is not None  # loop always runs ≥1 iteration
    probs = (probs_sum / n).squeeze(0)[..., :h, :w]
    return probs.cpu().numpy()


def _watershed_instances(seg_pred, h_min=2.0):
    field_mask = seg_pred == 1
    if not field_mask.any():
        return np.zeros_like(seg_pred, dtype=np.int32)
    distance = distance_transform_edt(field_mask).astype(np.float32)
    seeds = h_maxima(distance * field_mask, h=h_min)
    markers, _ = cc_label(seeds)
    if markers.max() == 0:
        markers, _ = cc_label(field_mask)
    inst = watershed(-distance, markers=markers, mask=field_mask)
    return inst.astype(np.int32)


def _gt_instances(mask):
    field = (mask == 1).astype(np.uint8)
    inst, _ = cc_label(field)
    return inst.astype(np.int32)


def _load_task(ckpt, device):
    t = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return t.model.eval().to(device)


def _load_planet(country, pid):
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
    return x.float(), np.asarray(y).astype(np.uint8)


def _planet_rgb_for_window(country, pid, window):
    p = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(p) as src:
        rgb = src.read([3, 2, 1])
    return np.transpose(rgb, (1, 2, 0)).astype(np.float32) / PLANET_SR_SCALE


def _s2_rgb_for_window(country, pid, window):
    s2 = Path("data/ftw") / country / "s2_images" / f"window_{window}" / f"{pid}.tif"
    planet = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr = dst.crs, dst.transform
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
                dst_transform=dst_tr,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    return np.transpose(out, (1, 2, 0)).astype(np.float32) / 10000.0


def _predict_s2_inst_to_planet_grid(model_s2, country, pid, device):
    s2_a = Path("data/ftw") / country / "s2_images" / "window_a" / f"{pid}.tif"
    s2_b = Path("data/ftw") / country / "s2_images" / "window_b" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32)
        b_crs, b_tr = src_b.crs, src_b.transform
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32)
    x = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0))
    probs = _predict_probs_tta(model_s2, x, device, scale=S2_NORM_DIVISOR, tta=True)
    seg = probs.argmax(axis=0).astype(np.uint8)
    inst = _watershed_instances(seg)
    planet = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr, dst_h, dst_w = dst.crs, dst.transform, dst.height, dst.width
    out = np.zeros((dst_h, dst_w), dtype=np.int32)
    reproject(
        source=inst,
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
    # 7 rows — distinct from v6, dense smallholder mix.
    p.add_argument(
        "--rows",
        nargs="+",
        default=[
            "croatia:g10-3_00019_18:a",
            "croatia:g10-3_00017_14:a",
            "slovenia:g13_00038_8:a",
            "austria:g83_00030_3:a",
            "austria:g94_00022_14:a",
            "lithuania:g11_00030_15:a",
            "finland:g14-1_00125_10:a",
        ],
    )
    p.add_argument("--out", default="paper/figs/qualitative_v7.pdf")
    p.add_argument("--cell-size", type=int, default=SQUARE_SIZE)
    p.add_argument("--cell-h", type=float, default=1.38)
    p.add_argument("--cell-w", type=float, default=1.35)
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
        print(f"  {country}:{pid} ({window})")
        x8, gt_full = _load_planet(country, pid)
        rgb_pl = _planet_rgb_for_window(country, pid, window)
        rgb_s2 = _s2_rgb_for_window(country, pid, window)
        probs_pl = _predict_probs_tta(model_pl, x8, device, scale=PLANET_SR_SCALE, tta=True)
        seg_pl = probs_pl.argmax(axis=0).astype(np.uint8)
        inst_pl = _watershed_instances(seg_pl)
        inst_s2 = _predict_s2_inst_to_planet_grid(model_s2, country, pid, device)
        inst_gt = _gt_instances(gt_full)
        sq = args.cell_size
        rgb_pl_s = _resize_nn(_stretch(_square_crop(rgb_pl)), sq)
        rgb_s2_s = _resize_nn(_stretch(_square_crop(rgb_s2)), sq)
        inst_gt_s = _resize_nn(_square_crop(inst_gt), sq)
        inst_pl_s = _resize_nn(_square_crop(inst_pl), sq)
        inst_s2_s = _resize_nn(_square_crop(inst_s2), sq)
        print(
            f"    instances: gt={int(inst_gt.max())} planet={int(inst_pl.max())} s2={int(inst_s2.max())}"
        )
        rows.append((country, pid, window, rgb_pl_s, rgb_s2_s, inst_gt_s, inst_pl_s, inst_s2_s))

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
        "GT instances",
        "PRUE-HD-B3 (ours)",
        "PRUE-B7 (S2 baseline)",
    ]
    for i, (country, pid, window, rgb_pl, rgb_s2, igt, ipl, is2) in enumerate(rows):
        axes[i, 0].imshow(rgb_pl)
        axes[i, 1].imshow(rgb_s2)
        axes[i, 2].imshow(_instance_render(igt))
        axes[i, 3].imshow(_instance_render(ipl))
        axes[i, 4].imshow(_instance_render(is2))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color("#444444")
        axes[i, 0].set_ylabel(country.replace("_", " "), fontsize=7.5, fontweight="bold")
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=8.5, fontweight="bold", pad=3)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
