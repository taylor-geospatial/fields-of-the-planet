"""Precompute spatially-aligned Planet TIFs.

For each Planet patch that has a corresponding S2 chip, reprojects window_a
and window_b to the S2 chip's CRS+bounds at Planet's native GSD (~3 m),
reorders bands from PSScene BGR(NIR) to RGB(NIR), and writes the result as
a compressed GeoTIFF.

Output layout (mirroring the original planet/ tree):
    <planet_root>/aligned_window_a/<country>/<patch_id>.tif
    <planet_root>/aligned_window_b/<country>/<patch_id>.tif

These files can be read with a plain ``rasterio.open / src.read()`` during
training — no on-the-fly warp needed.

Usage:
    uv run python scripts/precompute_aligned_planet.py \\
        --planet-root data/planet \\
        --s2-root data/ftw \\
        --workers 64
"""

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.coords
from rasterio.transform import from_bounds as _from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

# PSScene native band order: [B, G, R, NIR] → reindex to [R, G, B, NIR]
_BGR_TO_RGB = [2, 1, 0, 3]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _warp_window(
    src_path: str,
    dst_path: str,
    s2_crs: rasterio.crs.CRS,
    s2_bounds: rasterio.coords.BoundingBox,
) -> bool:
    """Reproject one Planet window to S2 CRS+bounds at native GSD.

    Returns True on success, False on recoverable error (logged).
    """
    try:
        with rasterio.open(src_path) as src:
            native_xform, _, _ = calculate_default_transform(
                src.crs,
                s2_crs,
                src.width,
                src.height,
                *src.bounds,
            )
            native_gsd = abs(native_xform.a)
            out_w = max(1, round((s2_bounds.right - s2_bounds.left) / native_gsd))
            out_h = max(1, round((s2_bounds.top - s2_bounds.bottom) / native_gsd))
            out_transform = _from_bounds(
                s2_bounds.left,
                s2_bounds.bottom,
                s2_bounds.right,
                s2_bounds.top,
                out_w,
                out_h,
            )
            out = np.zeros((src.count, out_h, out_w), dtype=np.float32)
            for band_i in range(src.count):
                reproject(
                    source=rasterio.band(src, band_i + 1),
                    destination=out[band_i],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=out_transform,
                    dst_crs=s2_crs,
                    resampling=Resampling.bilinear,
                )
            # Reorder BGR(N) → RGB(N)
            out = out[_BGR_TO_RGB]
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "width": out_w,
                "height": out_h,
                "count": src.count,
                "crs": s2_crs,
                "transform": out_transform,
                "compress": "deflate",
                "predictor": 3,  # floating-point predictor
                "zlevel": 6,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(out)
    except Exception as exc:  # worker — log and continue
        log.warning("FAILED %s: %s", src_path, exc)
        return False
    return True


def _process_patch(args: tuple) -> tuple[str, str, bool]:
    """Worker entry point. Returns (patch_id, country, success)."""
    (
        patch_id,
        country,
        planet_a,
        planet_b,
        s2_ref,
        out_a,
        out_b,
        skip_existing,
    ) = args

    need_a = not (skip_existing and Path(out_a).exists())
    need_b = not (skip_existing and Path(out_b).exists())
    if not need_a and not need_b:
        return patch_id, country, True

    try:
        with rasterio.open(s2_ref) as s2:
            s2_crs = s2.crs
            s2_bounds = s2.bounds
    except Exception as exc:
        log.warning("Cannot open S2 ref %s: %s", s2_ref, exc)
        return patch_id, country, False

    ok = True
    if need_a:
        ok &= _warp_window(planet_a, out_a, s2_crs, s2_bounds)
    if need_b:
        ok &= _warp_window(planet_b, out_b, s2_crs, s2_bounds)
    return patch_id, country, ok


def build_work_list(
    planet_root: Path,
    s2_root: Path,
    usable_only: bool,
    skip_existing: bool,
) -> list[tuple]:
    import geopandas as gpd

    idx = gpd.read_parquet(planet_root / "index.parquet")
    if usable_only:
        idx = idx[idx["usable_pair"] == True]  # noqa: E712
    idx["patch_id"] = idx["patch_id"].astype(str)

    work = []
    skipped_no_s2 = 0
    for row in idx.to_dict(orient="records"):
        country = row["country"]
        pid = row["patch_id"]
        s2_ref = s2_root / country / "s2_images" / "window_b" / f"{pid}.tif"
        if not s2_ref.exists():
            skipped_no_s2 += 1
            continue
        work.append(
            (
                pid,
                country,
                str(planet_root / row["image_a_path"]),
                str(planet_root / row["image_b_path"]),
                str(s2_ref),
                str(planet_root / "aligned_window_a" / country / f"{pid}.tif"),
                str(planet_root / "aligned_window_b" / country / f"{pid}.tif"),
                skip_existing,
            )
        )

    log.info("Patches to process: %d  (skipped no-S2: %d)", len(work), skipped_no_s2)
    return work


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planet-root", default="data/planet", type=Path)
    ap.add_argument("--s2-root", default="data/ftw", type=Path)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument(
        "--no-usable-filter",
        action="store_true",
        help="Include non-usable pairs (default: usable_only)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Re-compute even if output TIF already exists"
    )
    args = ap.parse_args()

    work = build_work_list(
        args.planet_root,
        args.s2_root,
        usable_only=not args.no_usable_filter,
        skip_existing=not args.overwrite,
    )
    if not work:
        log.info("Nothing to do.")
        sys.exit(0)

    log.info("Launching pool with %d workers", args.workers)
    n_ok = n_fail = 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (_pid, _country, ok) in enumerate(pool.imap_unordered(_process_patch, work), 1):
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if i % 500 == 0 or i == len(work):
                log.info("Progress: %d/%d  ok=%d  fail=%d", i, len(work), n_ok, n_fail)

    log.info("Done. ok=%d  fail=%d", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
