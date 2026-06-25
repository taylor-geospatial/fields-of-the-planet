"""Figure: held-out patches where higher resolution helps most.

Joins the two per-patch metric CSVs (Planet B3, S2 B7) on (country, patch_id),
ranks by ``delta_obj_f1 = planet_obj_f1 - s2_obj_f1``, picks the top patches
with delta>0 and enough GT fields (``--min-n-gt``), and renders one row per
patch:

  S2 RGB (10 m -> Planet grid) | Planet RGB (3 m) |
  GT mask | GT polygons | S2 mask | S2 polygons | Planet mask | Planet polygons

Each entity (GT, each model) is shown as its instance mask AND the polygons
vectorized from that exact mask. Each row is annotated with the country and the
two Obj F1 values + delta.

The GT polygons column draws the ACTUAL FTW vector parcels
(``clip_polygons_per_patch.py``, mapped from UTM into the display frame), not the
rasterized mask. The prediction polygon columns vectorize the post-processed
(TTA + watershed) field mask ``inst > 0`` with ``rasterio.features.shapes``
(GDAL polygonize / 4-connected components, the *same* extraction the PQ metric
scores), unsmoothed -- the staircase edges are the literal vectorizer output.

Both models are re-run here (TTA + watershed) using the exact eval functions
so the rendered polygons match the CSV metrics. Needs a GPU -> run via
``hpc/per_patch_eval.sbatch``-style sbatch, never the login node.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tg_style
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from rasterio.features import shapes as rio_shapes
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from shapely.affinity import affine_transform
from shapely.geometry import shape as shapely_shape

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
MASK_BG = np.array(mpl.colors.to_rgb(tg_style.BROWN), dtype=np.float32)  # dark brand brown


def _stretch(rgb, divisor=3000.0):
    """Constant-divisor reflectance stretch to [0, 1]: the same divisor on every
    channel preserves true color balance (matches hero.py). Both sensors store
    surface reflectance scaled by 1e4 (uint16)."""
    return np.clip(rgb.astype(np.float32) / divisor, 0.0, 1.0)


def _field_render(inst):
    """2-class field/background view (no per-instance colors): field interior
    green, background MASK_BG -- the mask columns show segmentation, the polygon
    columns carry the per-instance colors."""
    field = np.asarray(inst) > 0
    out = np.broadcast_to(MASK_BG, field.shape + (3,)).copy()
    out[field] = np.array(mpl.colors.to_rgb(tg_style.GREEN), dtype=np.float32)
    return out


def _polygonize_field_mask(field):
    """Connected-component polygons of a binary field mask via
    ``rasterio.features.shapes`` (GDAL polygonize, 4-connected) -- the exact
    extraction the PQ metric scores. Pass ``inst > 0`` for predictions or
    ``gt == 1`` for ground truth; adjacent fields vectorize as distinct polygons.
    """
    binary = np.asarray(field).astype(np.uint8)
    geoms = [shapely_shape(g) for g, v in rio_shapes(binary) if v == 1]
    return gpd.GeoDataFrame(geometry=geoms)


def _plot_polys(ax, geoms, size):
    """Draw field polygons (Glasbey fill, bold brand-brown outline) on white."""
    geoms = list(geoms)
    ax.set_facecolor("white")
    if geoms:
        colors = [tuple(c) for c in tg_style.glasbey_colors(len(geoms))]
        gpd.GeoDataFrame(geometry=geoms).plot(
            ax=ax, color=colors, edgecolor=tg_style.BROWN, linewidth=0.7
        )
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")


def _draw_field_polygons(ax, field, size):
    """Vectorize a binary field mask (connected components) and draw it."""
    _plot_polys(ax, _polygonize_field_mask(field).geometry, size)


def _true_gt_polys_display(country, pid, window, full_h, full_w, sq, root="data"):
    """Actual FTW field polygons for a patch, mapped into the square-cropped,
    resized display frame -- the real vector parcels, not the rasterized mask.
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


def _square_crop(arr):
    """Center-crop along the longer axis so output is square."""
    h, w = arr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return arr[y0 : y0 + side, x0 : x0 + side]


def _resize_square(arr, size):
    """Square-crop then resize to (size, size). Integer label maps use NEAREST
    (preserve instance ids); float RGB uses bilinear.

    The Planet patches are portrait (~518x350), so without this the predicted
    instance maps and RGB render as rectangles. Square-cropping + resizing here
    makes every panel a uniform square, matching the qualitative figures.
    """
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
    return np.transpose(rgb, (1, 2, 0)).astype(np.float32)  # raw DN; _stretch clips DN/3000


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
    return np.transpose(out, (1, 2, 0)).astype(np.float32)  # raw DN; _stretch clips DN/3000


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
    p.add_argument(
        "--sq-size", type=int, default=512, help="Square-crop+resize each panel to this."
    )
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

    sq = args.sq_size
    rows = []
    for _, r in sel.iterrows():
        country, pid = r["country"], str(r["patch_id"])
        print(f"  rendering {country}:{pid}")
        rgb_s2 = _stretch(_s2_rgb_on_planet_grid(country, pid, args.window))
        rgb_pl = _stretch(_planet_rgb(country, pid, args.window))
        inst_pl, gt_np = _predict_instances_planet(task_pl, model_pl, country, pid)
        inst_s2 = _predict_instances_s2_on_planet_grid(task_s2, model_s2, country, pid)
        gt_inst = _gt_instances(gt_np)
        gt_polys = _true_gt_polys_display(
            country, pid, args.window, gt_np.shape[0], gt_np.shape[1], sq
        )
        rows.append(
            {
                "country": country,
                "pid": pid,
                "rgb_s2": _resize_square(rgb_s2, sq),
                "rgb_pl": _resize_square(rgb_pl, sq),
                "gt": _resize_square(gt_inst, sq),
                "gt_polys": gt_polys,
                "inst_s2": _resize_square(inst_s2, sq),
                "inst_pl": _resize_square(inst_pl, sq),
                "f1_pl": r["obj_f1_pl"],
                "f1_s2": r["obj_f1_s2"],
                "delta": r["delta_obj_f1"],
            }
        )

    n = len(rows)
    cols = 8
    # Each entity (GT, each model) shows its instance mask AND its polygons.
    col_titles = [
        "Sentinel-2 (10 m)",
        "PlanetScope (3 m)",
        "GT mask",
        "GT polygons",
        "S2 mask\nFTW-PRUE+",
        "S2 polygons\nFTW-PRUE+",
        "Planet mask\nFTP-PRUE+",
        "Planet polygons\nFTP-PRUE+",
    ]
    fig, axes = plt.subplots(
        n, cols, figsize=(cols * 1.5, n * 1.62), gridspec_kw={"wspace": 0.02, "hspace": 0.05}
    )
    if n == 1:
        axes = axes[None, :]

    for i, row in enumerate(rows):
        axes[i, 0].imshow(row["rgb_s2"])
        axes[i, 1].imshow(row["rgb_pl"])
        axes[i, 2].imshow(_field_render(row["gt"]))
        _plot_polys(axes[i, 3], row["gt_polys"], row["gt"].shape[0])
        axes[i, 4].imshow(_field_render(row["inst_s2"]))
        _draw_field_polygons(axes[i, 5], row["inst_s2"] > 0, row["inst_s2"].shape[0])
        axes[i, 6].imshow(_field_render(row["inst_pl"]))
        _draw_field_polygons(axes[i, 7], row["inst_pl"] > 0, row["inst_pl"].shape[0])
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
        axes[i, 0].set_ylabel(
            label,
            fontsize=6.6,
            fontweight="bold",
            linespacing=1.25,
            rotation=0,
            ha="right",
            va="center",
            labelpad=44,
        )
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
