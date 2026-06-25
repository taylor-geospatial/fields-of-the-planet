"""Qualitative v8: raw-mask + vectorized field-polygon comparison.

8 columns x N rows -- each entity shown as a mask AND its vectorized polygons:
  Planet RGB | S2 RGB | GT mask | GT polygons |
  FTP-PRUE mask | FTP-PRUE polygons | FTW-PRUE (B7) mask | FTW-PRUE (B7) polygons

Same square-cropped, season-matched layout as v6/v7.  GT and prediction masks
use independent random RGB colors per field on a white background.  The GT *polygons* column
draws the ACTUAL FTW vector parcels (``clip_polygons_per_patch.py``, mapped from
UTM into the display frame) -- not the rasterized mask polygonized.  The
prediction polygon cells vectorize the adjacent mask with
``rasterio.features.shapes`` (GDAL polygonize / 4-connected components, the
*same* extraction the PQ metric scores), ``inst > 0``, unsmoothed -- so each
predicted mask sits beside its literal polygonization and is read against the
true GT parcels; 3 m vs 10 m fidelity is directly visible.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tg_style
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from rasterio.features import shapes as rio_shapes
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from shapely.affinity import affine_transform
from shapely.geometry import shape as shapely_shape
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

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
SQUARE_SIZE = 256
MASK_BG = np.array(mpl.colors.to_rgb(tg_style.BROWN), dtype=np.float32)  # dark brand brown


def _stretch(rgb, divisor=3000.0):
    """Constant-divisor reflectance stretch to [0, 1]: the same divisor on every
    channel preserves true color balance (matches hero.py). Both sensors store
    surface reflectance scaled by 1e4 (uint16)."""
    return np.clip(rgb.astype(np.float32) / divisor, 0.0, 1.0)


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


def _field_render(inst):
    """2-class field/background view (no per-instance colors): field interior
    green on MASK_BG -- the mask columns show segmentation, the polygon columns
    carry the per-instance colors."""
    field = np.asarray(inst) > 0
    out = np.broadcast_to(MASK_BG, field.shape + (3,)).copy()
    out[field] = np.array(mpl.colors.to_rgb(tg_style.GREEN), dtype=np.float32)
    return out


def _polygonize_field_mask(field):
    """Connected-component polygons of a binary field mask.

    ``rasterio.features.shapes`` is GDAL polygonize (4-connected components) --
    the exact extraction the PQ metric scores
    (``polygon_metrics_eval._extract_shapes``). Pass ``inst > 0`` for
    post-processed predictions (watershed already inserted 1-px gaps between
    adjacent parcels) or ``gt == 1`` for ground truth (the boundary class
    separates parcels). Either way adjacent fields vectorize as distinct polygons.
    """
    binary = np.asarray(field).astype(np.uint8)
    geoms = [shapely_shape(g) for g, v in rio_shapes(binary) if v == 1]
    return gpd.GeoDataFrame(geometry=geoms)


def _plot_polys(ax, geoms, size):
    """Draw field polygons on a white background in the square pixel frame
    (y increasing downward) that matches the imshow cells."""
    geoms = list(geoms)
    ax.set_facecolor("white")
    if geoms:
        colors = [tuple(c) for c in tg_style.glasbey_colors(len(geoms))]
        gpd.GeoDataFrame(geometry=geoms).plot(
            ax=ax, color=colors, edgecolor=tg_style.BROWN, linewidth=0.5
        )
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")


def _draw_field_polygons(ax, field, size):
    """Vectorize a binary field mask (connected components) and draw it."""
    _plot_polys(ax, _polygonize_field_mask(field).geometry, size)


def _true_gt_polys_display(country, pid, window, full_h, full_w, sq, root="data"):
    """Actual FTW field polygons for a patch, mapped into the square-cropped,
    resized display frame -- the real vector parcels, not the rasterized GT mask.

    Reads the per-patch GeoParquet from ``clip_polygons_per_patch.py`` (Planet
    UTM CRS), applies the inverse Planet ``window_a`` affine (UTM -> full pixel
    grid), then the same center square-crop + resize the image cells use.
    """
    ppath = Path(root) / "ftw_polygons_clipped" / country / f"{pid}.parquet"
    if not ppath.exists():
        return []
    tif = Path(root) / "planet" / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(tif) as src:
        inv = ~src.transform
    utm_to_px = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]
    side = min(full_h, full_w)
    x0, y0 = (full_w - side) // 2, (full_h - side) // 2
    s = sq / side
    crop_resize = [s, 0, 0, s, -x0 * s, -y0 * s]
    geoms = []
    for g in gpd.read_parquet(ppath).geometry:
        if g is None or g.is_empty:
            continue
        geoms.append(affine_transform(affine_transform(g, utm_to_px), crop_resize))
    return geoms


def _pad32(x, value=0.0, min_size=512):
    h, w = x.shape[-2], x.shape[-1]
    nh = max(((h + 31) // 32) * 32, min_size)
    nw = max(((w + 31) // 32) * 32, min_size)
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def _d4_transforms():
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
def _predict_raw_and_tta(model, x, device, scale):
    """Return (raw_argmax_mask, tta_argmax_mask) as uint8 HxW arrays."""
    x = (x.float() / scale).unsqueeze(0).to(device)
    xp, h, w = _pad32(x, min_size=512)
    # raw
    raw_logits = model(xp)
    raw = raw_logits.argmax(dim=1).squeeze(0)[..., :h, :w].cpu().numpy().astype(np.uint8)
    # TTA
    probs_sum = None
    n = 0
    for fwd, inv in _d4_transforms():
        out = model(fwd(xp))
        p = torch.softmax(out, dim=1)
        p = inv(p)
        probs_sum = p if probs_sum is None else probs_sum + p
        n += 1
    assert probs_sum is not None  # loop always runs ≥1 iteration
    tta_probs = (probs_sum / n).squeeze(0)[..., :h, :w]
    tta = tta_probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    return raw, tta


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
    return np.transpose(rgb, (1, 2, 0)).astype(np.float32)  # raw DN; _stretch clips DN/3000


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
    return np.transpose(out, (1, 2, 0)).astype(np.float32)  # raw DN; _stretch clips DN/3000


def _predict_s2_raw_inst_to_planet_grid(model_s2, country, pid, device):
    """Corrected resize_factor=2 protocol: bilinear-upsample the stacked 256
    S2 input to 512 before raw/TTA inference + watershed, then map the 512 raw
    mask and instance map onto the Planet grid via a transform scaled by
    256/512."""
    s2_a = Path("data/ftw") / country / "s2_images" / "window_a" / f"{pid}.tif"
    s2_b = Path("data/ftw") / country / "s2_images" / "window_b" / f"{pid}.tif"
    with rasterio.open(s2_b) as src_b:
        b_arr = src_b.read().astype(np.float32)
        b_crs, b_tr, bh, bw = src_b.crs, src_b.transform, src_b.height, src_b.width
    with rasterio.open(s2_a) as src_a:
        a_arr = src_a.read().astype(np.float32)
    x = torch.from_numpy(np.concatenate([b_arr, a_arr], axis=0)).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(S2_UPSAMPLE, S2_UPSAMPLE), mode="bilinear", align_corners=False)
    raw_s2, tta_s2 = _predict_raw_and_tta(model_s2, x.squeeze(0), device, scale=S2_NORM_DIVISOR)
    inst_s2 = _watershed_instances(tta_s2)
    up_tr = b_tr * rasterio.Affine.scale(bw / S2_UPSAMPLE, bh / S2_UPSAMPLE)
    planet = Path("data/planet") / country / "window_a" / f"{pid}.tif"
    with rasterio.open(planet) as dst:
        dst_crs, dst_tr, dst_h, dst_w = dst.crs, dst.transform, dst.height, dst.width
    # Reproject the raw mask (uint8) and the instance map (int32) to Planet grid.
    raw_out = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=raw_s2,
        destination=raw_out,
        src_transform=up_tr,
        src_crs=b_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    inst_out = np.zeros((dst_h, dst_w), dtype=np.int32)
    reproject(
        source=inst_s2,
        destination=inst_out,
        src_transform=up_tr,
        src_crs=b_crs,
        dst_transform=dst_tr,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return raw_out, inst_out


def _gt_instances(mask):
    field = (mask == 1).astype(np.uint8)
    inst, _ = cc_label(field)
    return inst.astype(np.int32)


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
    p.add_argument(
        "--rows",
        nargs="+",
        default=[
            "lithuania:g11_00062_5:a",
            "latvia:g31_00052_15:a",
            "belgium:g2_00021_11:a",
            "lithuania:g8_00052_15:a",
        ],
    )
    p.add_argument("--out", default="paper/figs/qualitative_main.pdf")
    p.add_argument("--cell-size", type=int, default=SQUARE_SIZE)
    p.add_argument("--cell-h", type=float, default=1.05)
    p.add_argument("--cell-w", type=float, default=1.05)
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
        # Planet: TTA argmax -> watershed instances.
        _, tta_pl = _predict_raw_and_tta(model_pl, x8, device, scale=PLANET_SR_SCALE)
        inst_pl = _watershed_instances(tta_pl)
        # S2: watershed instances reprojected to Planet grid.
        _, inst_s2 = _predict_s2_raw_inst_to_planet_grid(model_s2, country, pid, device)
        inst_gt = _gt_instances(gt_full)
        sq = args.cell_size
        rgb_pl_s = _resize_nn(_stretch(_square_crop(rgb_pl)), sq)
        rgb_s2_s = _resize_nn(_stretch(_square_crop(rgb_s2)), sq)
        gt_s = _resize_nn(_square_crop(inst_gt), sq)
        inst_pl_s = _resize_nn(_square_crop(inst_pl), sq)
        inst_s2_s = _resize_nn(_square_crop(inst_s2), sq)
        gt_polys = _true_gt_polys_display(
            country, pid, window, gt_full.shape[0], gt_full.shape[1], sq
        )
        print(
            f"    instances: gt={int(inst_gt.max())} planet={int(inst_pl.max())} "
            f"s2={int(inst_s2.max())} | true gt polys={len(gt_polys)}"
        )
        rows.append(
            (
                country,
                pid,
                window,
                rgb_pl_s,
                rgb_s2_s,
                gt_s,
                gt_polys,
                inst_pl_s,
                inst_s2_s,
            )
        )

    n = len(rows)
    cols = 8
    _fig, axes = plt.subplots(
        n,
        cols,
        figsize=(cols * args.cell_w, n * args.cell_h),
        gridspec_kw={"wspace": 0.015, "hspace": 0.03},
    )
    if n == 1:
        axes = axes[None, :]
    # Each entity (GT, each model) shows its post-processed mask AND the polygons
    # vectorized from that exact mask.
    col_titles = [
        "Input\nPlanet RGB",
        "Input\nS2 RGB",
        "GT\nmask",
        "GT\npolygons",
        "FTP-PRUE+\nmask",
        "FTP-PRUE+\npolygons",
        "FTW-PRUE+\nmask",
        "FTW-PRUE+\npolygons",
    ]
    for i, (
        country,
        pid,
        window,
        rgb_pl,
        rgb_s2,
        gt,
        gt_polys,
        inst_pl,
        inst_s2,
    ) in enumerate(rows):
        axes[i, 0].imshow(rgb_pl)
        axes[i, 1].imshow(rgb_s2)
        axes[i, 2].imshow(_field_render(gt))
        _plot_polys(axes[i, 3], gt_polys, gt.shape[0])
        axes[i, 4].imshow(_field_render(inst_pl))
        _draw_field_polygons(axes[i, 5], inst_pl > 0, inst_pl.shape[0])
        axes[i, 6].imshow(_field_render(inst_s2))
        _draw_field_polygons(axes[i, 7], inst_s2 > 0, inst_s2.shape[0])
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color(tg_style.BROWN)
        axes[i, 0].set_ylabel(country.replace("_", " "), fontsize=7.5, fontweight="bold")
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=7.5, fontweight="bold", pad=3, linespacing=1.05)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
