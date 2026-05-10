"""Render hero figure: side-by-side FTW S2, PlanetScope SR, and label for several patches.

Picks one representative patch per country across a diverse country set,
loads (Sentinel-2 RGB, PlanetScope RGB, 3-class label), reprojects S2 onto
the Planet UTM grid for visual comparison, and writes a publication PDF.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject

# Order matters — left column appears first in the figure.
COUNTRIES = ("rwanda", "denmark", "france", "brazil", "india")

# FTW S2 chip band order: [B04 (R), B03 (G), B02 (B), B08 (NIR)] -> RGB = [1,2,3]
# Planet SR band order:   [Blue, Green, Red, NIR]              -> RGB = [3,2,1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ftw-root", type=Path, default=Path("../data/ftw"))
    p.add_argument("--planet-root", type=Path, default=Path("../data/planet"))
    p.add_argument("--out", type=Path, default=Path("hero.pdf"))
    p.add_argument("--countries", nargs="+", default=list(COUNTRIES))
    return p.parse_args()


def _percentile_stretch(rgb: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Per-channel percentile stretch -> [0,1] float."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[0]):
        v = rgb[c].astype(np.float32)
        lo, hi = np.percentile(v[v > 0], [p_low, p_high]) if (v > 0).any() else (0, 1)
        if hi <= lo:
            hi = lo + 1
        out[c] = np.clip((v - lo) / (hi - lo), 0, 1)
    return out


def _load_planet_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        b = src.read([3, 2, 1])  # Red, Green, Blue
        rgb = _percentile_stretch(b)
    return np.transpose(rgb, (1, 2, 0))


def _load_s2_on_planet_grid(s2_path: Path, planet_path: Path) -> np.ndarray:
    """Read S2 RGB and reproject to the Planet patch's UTM grid for visual comparison."""
    with rasterio.open(planet_path) as dst:
        dst_crs, dst_transform = dst.crs, dst.transform
        dst_h, dst_w = dst.height, dst.width
    with rasterio.open(s2_path) as src:
        bands = src.read([1, 2, 3])  # Red, Green, Blue (FTW chip)
        out = np.zeros((3, dst_h, dst_w), dtype=bands.dtype)
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
    rgb = _percentile_stretch(out)
    return np.transpose(rgb, (1, 2, 0))


def _load_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _pick_patch(country: str, planet_root: Path) -> tuple[str, str] | None:
    """Find a (patch_id, window) where Planet SR + UDM2 + label all exist."""
    pdir = planet_root / country
    if not pdir.is_dir():
        return None
    for sr in sorted(pdir.glob("*_a.tif")):
        if "_udm2" in sr.name or "_label" in sr.name:
            continue
        stem = sr.stem  # e.g. 1592589_a
        label = pdir / f"{stem}_label.tif"
        if label.exists():
            return stem.rsplit("_", 1)[0], "a"
    return None


def main() -> int:
    args = parse_args()
    n = len(args.countries)
    _fig, axes = plt.subplots(n, 3, figsize=(7, 2.4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    label_cmap = ListedColormap(["#000000", "#79C753", "#FFD23F"])

    for r, country in enumerate(args.countries):
        pick = _pick_patch(country, args.planet_root)
        if not pick:
            for c in range(3):
                axes[r, c].axis("off")
                axes[r, c].set_title(f"({country}: no sample)", fontsize=8)
            continue
        pid, win = pick
        planet_sr = args.planet_root / country / f"{pid}_{win}.tif"
        planet_lbl = args.planet_root / country / f"{pid}_{win}_label.tif"
        s2 = args.ftw_root / country / "s2_images" / f"window_{win}" / f"{pid}.tif"

        rgb_s2 = _load_s2_on_planet_grid(s2, planet_sr)
        rgb_pl = _load_planet_rgb(planet_sr)
        lbl = _load_label(planet_lbl)

        for c, (img, kw) in enumerate(
            [
                (rgb_s2, {}),
                (rgb_pl, {}),
                (lbl, {"cmap": label_cmap, "vmin": 0, "vmax": 2, "interpolation": "nearest"}),
            ]
        ):
            ax = axes[r, c]
            if img.ndim == 3:
                ax.imshow(img, **kw)
            else:
                ax.imshow(img, **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(
                    ["FTW Sentinel-2 (10 m)", "PlanetScope (3 m)", "Label (3-class)"][c], fontsize=9
                )
            if c == 0:
                ax.set_ylabel(country.replace("_", " ").title(), fontsize=8)

    plt.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3)
    plt.savefig(args.out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
