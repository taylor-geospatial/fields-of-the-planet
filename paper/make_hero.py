"""Hero gallery: 12 high-quality FTW patches as (S2, Planet, label) triplets.

Picks patches with clear UDM2 stats (clear >= 0.99) drawn from a diverse
country set, reprojects the FTW Sentinel-2 chip to the matched Planet
patch's UTM grid, resamples every image to square at a fixed size, and
tiles 12 triplets into a wide 6x6 layout (each row carries 2 triplets =
6 cells side-by-side).

Reflectance normalisation:
  * Sentinel-2: 16-bit reflectance scaled by 10,000 (FTW convention).
    Divide by 3000 and clip to [0, 1] -> emphasises field structure
    without crushing bright soil.
  * PlanetScope: 16-bit surface reflectance scaled by 10,000.
    Same normaliser applies; vegetation/soil reflectance maxes out around
    30%, so dividing by 3000 lands the bulk in (0, 1).
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject
from skimage.transform import resize

# FTW S2 band order: [B04 (R), B03 (G), B02 (B), B08 (NIR)] -> RGB = (1, 2, 3)
# Planet 4-band SR:  [Blue, Green, Red, NIR]              -> RGB = (3, 2, 1)
NORM_DIVISOR = 3000.0
SQUARE_PX = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ftw-root", type=Path, default=Path("../data/ftw"))
    p.add_argument("--planet-root", type=Path, default=Path("../data/planet"))
    p.add_argument(
        "--udm2-quality", type=Path, default=Path("../data/planet/_global/udm2_quality.jsonl")
    )
    p.add_argument("--out", type=Path, default=Path("hero.pdf"))
    p.add_argument("--n", type=int, default=12, help="Number of triplets to include.")
    p.add_argument(
        "--min-field-pct",
        type=float,
        default=0.40,
        help="Minimum fraction of patch pixels that are field interior (class 1).",
    )
    p.add_argument(
        "--max-check",
        type=int,
        default=2000,
        help="Max candidates to score by field-fraction before stopping.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _stretch(rgb: np.ndarray, divisor: float = NORM_DIVISOR) -> np.ndarray:
    """Divide-and-clip normaliser. Inputs uint16 reflectance * 10000; output [0,1] float."""
    out = rgb.astype(np.float32) / divisor
    return np.clip(out, 0.0, 1.0)


def _to_square(img: np.ndarray, size: int = SQUARE_PX) -> np.ndarray:
    """Resample (H, W, C) or (H, W) to (size, size, C) or (size, size)."""
    order = 1 if img.ndim == 3 else 0
    return resize(
        img,
        (size, size) if img.ndim == 2 else (size, size, img.shape[-1]),
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0),
    )


def _load_planet_rgb(path: Path, size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        bgr_nir = src.read([3, 2, 1])  # Red, Green, Blue
    rgb = np.transpose(bgr_nir, (1, 2, 0))
    return _to_square(_stretch(rgb), size)


def _load_s2_on_planet_grid(s2_path: Path, planet_path: Path, size: int) -> np.ndarray:
    with rasterio.open(planet_path) as dst:
        dst_crs, dst_transform = dst.crs, dst.transform
        dst_h, dst_w = dst.height, dst.width
    with rasterio.open(s2_path) as src:
        bands = src.read([1, 2, 3])  # R, G, B from FTW chip
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
    rgb = np.transpose(out, (1, 2, 0))
    return _to_square(_stretch(rgb), size)


def _load_label(path: Path, size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        lbl = src.read(1)
    return _to_square(lbl, size).astype(np.uint8)


def _candidate_patches(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Filter UDM2 quality rows for high-clear patches with all three files on disk."""
    rows: list[dict] = []
    with args.udm2_quality.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = [r for r in rows if r.get("clear", 0) >= 0.99 and r.get("unusable", 1) <= 0.01]
    # Verify SR, label, and S2 all exist.
    out: list[tuple[str, str, str]] = []
    for r in rows:
        c, pid, w = r.get("country"), r.get("id"), r.get("window")
        if not (c and pid and w):
            continue
        sr = args.planet_root / c / f"{pid}_{w}.tif"
        lbl = args.planet_root / c / f"{pid}_{w}_label.tif"
        s2 = args.ftw_root / c / "s2_images" / f"window_{w}" / f"{pid}.tif"
        if sr.exists() and lbl.exists() and s2.exists():
            out.append((c, pid, w))
    return out


def _label_field_fraction(planet_root: Path, country: str, pid: str, window: str) -> float:
    """Return the fraction of label pixels that are class 1 (field interior)."""
    p = planet_root / country / f"{pid}_{window}_label.tif"
    try:
        with rasterio.open(p) as src:
            lbl = src.read(1)
        return float((lbl == 1).sum()) / lbl.size
    except Exception:
        return 0.0


def _filter_field_dense(
    cands: list[tuple[str, str, str]],
    planet_root: Path,
    min_field_pct: float,
    max_check: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    """Sample candidates and keep those with field coverage above threshold."""
    rng = random.Random(seed)
    rng.shuffle(cands)
    kept: list[tuple[float, tuple[str, str, str]]] = []
    for i, (c, pid, w) in enumerate(cands):
        if i >= max_check:
            break
        frac = _label_field_fraction(planet_root, c, pid, w)
        if frac >= min_field_pct:
            kept.append((frac, (c, pid, w)))
    # Return high-coverage first.
    kept.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in kept]


def _pick_diverse(
    candidates: list[tuple[str, str, str]], n: int, seed: int
) -> list[tuple[str, str, str]]:
    """Round-robin across countries to maximise diversity, then shuffle."""
    by_country: dict[str, list] = {}
    for c, pid, w in candidates:
        by_country.setdefault(c, []).append((c, pid, w))
    rng = random.Random(seed)
    for cs in by_country.values():
        rng.shuffle(cs)
    picks: list[tuple[str, str, str]] = []
    while len(picks) < n and by_country:
        for c in list(by_country):
            if not by_country[c]:
                del by_country[c]
                continue
            picks.append(by_country[c].pop())
            if len(picks) >= n:
                break
    return picks[:n]


def main() -> int:
    args = parse_args()
    cands = _candidate_patches(args)
    print(f"{len(cands)} candidate patches with clear>=0.99 and all files present")
    dense = _filter_field_dense(
        cands, args.planet_root, args.min_field_pct, args.max_check, args.seed
    )
    print(
        f"{len(dense)} candidates with field coverage >= {args.min_field_pct:.0%} "
        f"(checked up to {args.max_check} samples)"
    )
    picks = _pick_diverse(dense, args.n, args.seed)
    print(f"selected {len(picks)} triplets:")
    for c, pid, w in picks:
        print(f"  {c:14s} {pid}_{w}")

    # Layout: each row holds 2 triplets = 6 cells; with n=12 -> 6 rows
    triplets_per_row = 2
    n_rows = (len(picks) + triplets_per_row - 1) // triplets_per_row
    fig, axes = plt.subplots(
        n_rows,
        triplets_per_row * 3,
        figsize=(triplets_per_row * 3 * 1.4, n_rows * 1.4),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    label_cmap = ListedColormap(["#000000", "#79C753", "#FFD23F"])

    for r in range(n_rows):
        for t in range(triplets_per_row):
            idx = r * triplets_per_row + t
            base_col = t * 3
            for c in range(3):
                ax = axes[r, base_col + c]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
            if idx >= len(picks):
                for c in range(3):
                    axes[r, base_col + c].axis("off")
                continue
            country, pid, w = picks[idx]
            sr = args.planet_root / country / f"{pid}_{w}.tif"
            lbl = args.planet_root / country / f"{pid}_{w}_label.tif"
            s2 = args.ftw_root / country / "s2_images" / f"window_{w}" / f"{pid}.tif"
            rgb_s2 = _load_s2_on_planet_grid(s2, sr, SQUARE_PX)
            rgb_pl = _load_planet_rgb(sr, SQUARE_PX)
            lbl_img = _load_label(lbl, SQUARE_PX)

            axes[r, base_col + 0].imshow(rgb_s2)
            axes[r, base_col + 1].imshow(rgb_pl)
            axes[r, base_col + 2].imshow(
                lbl_img, cmap=label_cmap, vmin=0, vmax=2, interpolation="nearest"
            )
            if r == 0:
                axes[r, base_col + 0].set_title("S2 (10 m)", fontsize=8, pad=2)
                axes[r, base_col + 1].set_title("Planet (3 m)", fontsize=8, pad=2)
                axes[r, base_col + 2].set_title("Label", fontsize=8, pad=2)
            axes[r, base_col + 0].set_ylabel(country.replace("_", " "), fontsize=7)

    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.05)
    plt.savefig(args.out, bbox_inches="tight", dpi=140)
    plt.close()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
