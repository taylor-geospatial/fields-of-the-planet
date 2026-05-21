"""Coverage figure: actual PlanetScope RGB patches placed onto an Esri
World Imagery basemap at each patch's geographic location.

Shows the *sparse* nature of FTW sampling — the basemap covers the entire
country, but only a few hundred 3 m chips are illuminated as crisp colored
squares scattered across it.

For each requested country we:
  1. Collect every window_a/*.tif (one chip per patch_id).  These are in the
     PSScene native UTM grid.
  2. Reproject each chip to EPSG:3857 (Web Mercator) at a small thumbnail
     size (default 24x24 px) for cheap compositing.
  3. Render an Esri WorldImagery basemap covering the union extent.
  4. Plot every reprojected thumbnail at its geographic extent with
     ``imshow(..., extent=...)``.

Output: paper/figs/patches_on_basemap.pdf  (multi-country grid).
"""

import argparse
from pathlib import Path

import contextily as ctx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

WM = "EPSG:3857"
NORM_DIVISOR = 3000.0  # match the hero stretch
THUMB_PX = 64  # final reprojected size per patch


def _load_thumb(path: Path, dst_size: int = THUMB_PX):
    """Reproject one Planet patch SR tif to Web Mercator, return (rgb [0..1],
    extent [xmin, xmax, ymin, ymax]) in EPSG:3857 meters."""
    with rasterio.open(path) as src:
        src_bounds = src.bounds
        src_crs = src.crs
        # Read R, G, B from a 4-band BGR(+NIR) Dove product: band order is
        # Blue, Green, Red, NIR (per PSScene 4-band spec).  RGB = (3, 2, 1).
        bgr = src.read([3, 2, 1])
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs,
            WM,
            src.width,
            src.height,
            *src_bounds,
            dst_width=dst_size,
            dst_height=dst_size,
        )
        out = np.zeros((3, dst_size, dst_size), dtype=np.float32)
        for i in range(3):
            reproject(
                source=bgr[i].astype(np.float32),
                destination=out[i],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=WM,
                resampling=Resampling.bilinear,
            )
    rgb = np.transpose(out, (1, 2, 0))
    rgb = np.clip(rgb / NORM_DIVISOR, 0.0, 1.0)
    # Compute extent (xmin, xmax, ymin, ymax) in WM meters from the
    # destination transform.
    xmin = dst_transform.c
    ymax = dst_transform.f
    xmax = xmin + dst_w * dst_transform.a
    ymin = ymax + dst_h * dst_transform.e  # e is negative
    return rgb, (xmin, xmax, ymin, ymax)


def _country_patches(planet_root: Path, country: str, max_patches: int | None = None):
    """Yield all window_a SR tifs for a country (one per patch_id)."""
    fs = sorted((planet_root / country / "window_a").glob("*.tif"))
    if max_patches and len(fs) > max_patches:
        # Sample evenly across the file list so we keep geographic spread,
        # rather than just taking the first N (which would be one cluster).
        step = len(fs) / max_patches
        fs = [fs[int(i * step)] for i in range(max_patches)]
    return fs


def _render_country(
    ax,
    planet_root: Path,
    country: str,
    max_patches: int,
    title: str | None = None,
    basemap_zoom: int | None = None,
):
    files = _country_patches(planet_root, country, max_patches=max_patches)
    thumbs = []
    extents = []
    for i, f in enumerate(files):
        try:
            rgb, ext = _load_thumb(f)
        except Exception as err:
            print(f"  skip {f.name}: {err}")
            continue
        thumbs.append(rgb)
        extents.append(ext)
        if i and i % 250 == 0:
            print(f"    {country}: {i}/{len(files)}")
    if not thumbs:
        ax.set_axis_off()
        ax.set_title(f"{country} (no patches)", fontsize=9)
        return

    # Country-wide extent for the basemap, then add a small pad.
    xs = [e[0] for e in extents] + [e[1] for e in extents]
    ys = [e[2] for e in extents] + [e[3] for e in extents]
    pad_x = (max(xs) - min(xs)) * 0.04
    pad_y = (max(ys) - min(ys)) * 0.04
    extent_bm = (min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y)
    ax.set_xlim(extent_bm[0], extent_bm[1])
    ax.set_ylim(extent_bm[2], extent_bm[3])

    if basemap_zoom is None:
        extent_m = max(extent_bm[1] - extent_bm[0], extent_bm[3] - extent_bm[2])
        if extent_m < 2.5e5:
            basemap_zoom = 9
        elif extent_m < 1.5e6:
            basemap_zoom = 7
        else:
            basemap_zoom = 5
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,  # ty: ignore[unresolved-attribute]  # xyzservices stubs incomplete
            zoom=basemap_zoom,
            attribution=False,
        )
    except Exception as err:
        print(f"  basemap unavailable for {country}: {err}")

    # Each FTW chip is ~1.5 km on a side, which at country zoom is below
    # the pixel grid of the rendered figure.  We inflate every patch's
    # display extent around its centroid by `display_scale` so the chips
    # read as visible squares; this is a *visual exaggeration only* — true
    # geographic positions are preserved.  The caption flags this.
    display_scale = 6.0
    for rgb, ext in zip(thumbs, extents):
        xc = 0.5 * (ext[0] + ext[1])
        yc = 0.5 * (ext[2] + ext[3])
        dx = 0.5 * (ext[1] - ext[0]) * display_scale
        dy = 0.5 * (ext[3] - ext[2]) * display_scale
        ax.imshow(
            rgb,
            extent=(xc - dx, xc + dx, yc - dy, yc + dy),
            origin="upper",
            interpolation="bilinear",
            zorder=10,
        )
        # Thin outline helps the chip read against busy basemap regions
        # (forest canopy, cropland).
        ax.plot(
            [xc - dx, xc + dx, xc + dx, xc - dx, xc - dx],
            [yc - dy, yc - dy, yc + dy, yc + dy, yc - dy],
            color="white",
            linewidth=0.4,
            alpha=0.7,
            zorder=11,
        )

    n = len(thumbs)
    label = title or country.replace("_", " ").title()
    ax.set_title(f"{label}  ·  n={n:,} patches", fontsize=9.5, fontweight="bold", pad=4, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument(
        "--countries",
        nargs="+",
        default=["cambodia", "rwanda", "south_africa", "vietnam"],
    )
    p.add_argument(
        "--max-patches", type=int, default=500, help="Per-country patch cap (subsampled evenly)."
    )
    p.add_argument("--out", type=Path, default=Path("paper/figs/patches_on_basemap.pdf"))
    p.add_argument("--cols", type=int, default=2)
    args = p.parse_args()

    n = len(args.countries)
    rows = (n + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 4.2, rows * 4.0))
    axes = np.atleast_2d(axes).flatten()
    for ax, country in zip(axes, args.countries):
        print(f"rendering {country}...")
        _render_country(ax, args.planet_root, country, args.max_patches)
    for ax in axes[len(args.countries) :]:
        ax.set_axis_off()
    fig.suptitle(
        "FTW-HD covers each country with a sparse scatter of 3 m chips",
        fontsize=12,
        fontweight="bold",
        x=0.04,
        ha="left",
        y=0.985,
    )
    # Secondary line for the disclaimer about display-scale exaggeration.
    fig.text(
        0.04,
        0.962,
        "PlanetScope SR (RGB) overlaid on Esri World Imagery basemap. "
        "Chip footprints are exaggerated 6× for visibility; true centroids preserved.",
        fontsize=8.5,
        color="#555555",
        ha="left",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95), pad=0.6)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
