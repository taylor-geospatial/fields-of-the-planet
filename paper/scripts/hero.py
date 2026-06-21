"""Hero gallery: FTW patches as (S2, Planet, label) triplets for the page-1 hero.

Each triplet shows the FTW Sentinel-2 chip (10 m), the matched PlanetScope SR
image (3 m), and the 3-class field label, rendered as a 3-rows x 3-triplets
banner spanning both columns.

Geometry: FTW S2 chips are stored in EPSG:4326 while Planet patches are in
native UTM, so the S2 chip is reprojected onto the Planet grid. That reprojection
turns the lat/lon square into a slightly rotated quad with nodata in the corners,
so all three panels are cropped to the largest fully-valid centred square (which
also fixes the non-square patch aspect) before upsampling to a fixed size. The S2
chip is resampled nearest so its coarse 10 m pixels stay visibly blocky next to
the sharp 3 m Planet image; the label is nearest to preserve class ids.

Display uses a per-image 2-98 percentile stretch so scenes of varying brightness
(e.g. dark smallholder paddies vs bright European fields) all expose well. The
label uses the shared Taylor Geospatial palette (see ``tg_style``). No logos: the
paper is a double-blind submission.

The default triplets are curated from data/planet/index.parquet: recent (2021+),
UDM2-clear, high field-instance density, and geographically diverse (two
smallholder scenes for contrast, seven European fields). Override with --patches.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tg_style
from rasterio.warp import Resampling, reproject
from skimage.transform import resize

# Match the paper body face (Nimbus Roman = URW Times clone) for the column
# titles so they read as part of the document rather than a generic plot.
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times"]})

# FTW S2 band order [B04, B03, B02, B08] -> RGB = (1, 2, 3); Planet 4-band SR
# [Blue, Green, Red, NIR] -> RGB = (3, 2, 1).
SQUARE_PX = 512

# Both sensors store surface reflectance scaled by 1e4 (uint16). A single
# constant divisor preserves true color balance; the earlier per-channel
# percentile stretch normalized each band independently and shifted the
# white balance (the "looks BGR/oddly normalized" artifact).
NORM_DIVISOR = 3000.0

DEFAULT_PATCHES = (
    ("cambodia", "g33_0000000000-0000000000", "a"),
    ("croatia", "g10-3_00009_2", "a"),
    ("austria", "g95_00041_14", "a"),
    ("rwanda", "1592645", "a"),
    ("slovenia", "g13_00046_6", "a"),
    ("lithuania", "g11_00098_11", "a"),
    ("latvia", "g26_00059_12", "a"),
    ("denmark", "g6_00078_0", "a"),
    ("sweden", "g6-0_00064_5", "a"),
)


def _stretch(rgb: np.ndarray, divisor: float = NORM_DIVISOR) -> np.ndarray:
    """Constant-divisor reflectance stretch to [0, 1], same divisor on every
    channel so the true color balance is preserved."""
    return np.clip(rgb.astype(np.float32) / divisor, 0.0, 1.0)


def _largest_valid_square(valid: np.ndarray) -> tuple[int, int, int, int]:
    """Largest centred square fully inside the boolean ``valid`` mask.

    The reprojected S2 chip is a rotated quad with nodata corners; this crop
    removes the nodata while keeping the S2/Planet/label panels pixel-aligned.
    """
    ys, xs = np.nonzero(valid)
    cy, cx = int(ys.mean()), int(xs.mean())
    h, w = valid.shape
    lo, hi = 0, min(cy, h - cy, cx, w - cx)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if valid[cy - mid : cy + mid, cx - mid : cx + mid].all():
            lo = mid
        else:
            hi = mid - 1
    return cy - lo, cy + lo, cx - lo, cx + lo


def _resize(img: np.ndarray, order: int) -> np.ndarray:
    shape = (SQUARE_PX, SQUARE_PX) if img.ndim == 2 else (SQUARE_PX, SQUARE_PX, img.shape[-1])
    return resize(img, shape, order=order, preserve_range=True, anti_aliasing=(order > 0))


def _load_triplet(
    country: str, pid: str, window: str, planet_root: Path, ftw_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (s2_rgb, planet_rgb, label), all SQUARE_PX and cropped to the same
    valid square so the three panels stay spatially aligned."""
    sr = planet_root / country / f"window_{window}" / f"{pid}.tif"
    s2 = ftw_root / country / "s2_images" / f"window_{window}" / f"{pid}.tif"
    lbl_path = planet_root / country / "labels" / f"{pid}.tif"

    with rasterio.open(sr) as p:
        planet = np.transpose(p.read([3, 2, 1]), (1, 2, 0))
        dst_crs, dst_transform = p.crs, p.transform
        h, w = p.height, p.width
    with rasterio.open(lbl_path) as s:
        label = s.read(1)
    with rasterio.open(s2) as s:
        bands = s.read([1, 2, 3])
        s2_grid = np.zeros((3, h, w), dtype=bands.dtype)
        valid = np.zeros((h, w), dtype=np.float32)
        for i in range(3):
            reproject(
                source=bands[i],
                destination=s2_grid[i],
                src_transform=s.transform,
                src_crs=s.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        reproject(
            source=np.ones_like(bands[0]),
            destination=valid,
            src_transform=s.transform,
            src_crs=s.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    s2_grid = np.transpose(s2_grid, (1, 2, 0))

    y0, y1, x0, x1 = _largest_valid_square(valid > 0.5)
    s2_rgb = _resize(_stretch(s2_grid[y0:y1, x0:x1]), order=0)
    planet_rgb = _resize(_stretch(planet[y0:y1, x0:x1]), order=1)
    label_sq = _resize(label[y0:y1, x0:x1], order=0).astype(np.uint8)
    return s2_rgb, planet_rgb, label_sq


def _parse_patch(spec: str) -> tuple[str, str, str]:
    country, pid, window = spec.split(":")
    return (country, pid, window)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("../data/planet"))
    p.add_argument("--ftw-root", type=Path, default=Path("../data/ftw"))
    p.add_argument("--out", type=Path, default=Path("hero.pdf"))
    p.add_argument("--triplets-per-row", type=int, default=3)
    p.add_argument(
        "--patches",
        nargs="+",
        type=_parse_patch,
        default=list(DEFAULT_PATCHES),
        help="Triplets as country:patch_id:window; defaults to the curated set.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    picks = args.patches
    tpr = args.triplets_per_row
    n_rows = (len(picks) + tpr - 1) // tpr

    _fig, axes = plt.subplots(n_rows, tpr * 3, figsize=(tpr * 3 * 1.1, n_rows * 1.1 + 0.12))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    label_cmap = tg_style.label_cmap()

    for r in range(n_rows):
        for t in range(tpr):
            idx = r * tpr + t
            base = t * 3
            for c in range(3):
                ax = axes[r, base + c]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("black")
            if idx >= len(picks):
                for c in range(3):
                    axes[r, base + c].axis("off")
                continue
            country, pid, w = picks[idx]
            s2_rgb, planet_rgb, label = _load_triplet(
                country, pid, w, args.planet_root, args.ftw_root
            )
            axes[r, base + 0].imshow(s2_rgb)
            axes[r, base + 1].imshow(planet_rgb)
            axes[r, base + 2].imshow(
                label, cmap=label_cmap, vmin=0, vmax=2, interpolation="nearest"
            )
            if r == 0:
                tkw = {"fontsize": 10, "pad": 3, "color": tg_style.BROWN}
                axes[r, base + 0].set_title("S2 (10 m)", **tkw)
                axes[r, base + 1].set_title("Planet (3 m)", **tkw)
                axes[r, base + 2].set_title("Label", **tkw)
            print(f"{country:12s} {pid}_{w}")

    plt.tight_layout(pad=0.1, h_pad=0.0, w_pad=0.0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(args.out, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
