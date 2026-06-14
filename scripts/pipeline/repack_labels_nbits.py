"""Re-encode existing label tifs with NBITS=2 packed storage.

Labels are 3-class uint8 (0=background, 1=field, 2=boundary). Stored as
plain uint8 they use 8 bits/pixel before compression. Setting ``NBITS=2``
packs 4 pixels per byte at the bit level before ZSTD — the file stays a
standard GeoTIFF (rasterio auto-unpacks back to uint8 on read) but the
on-disk size shrinks roughly 2-3x further.

Idempotent: skips tifs whose GeoTIFF ``NBITS=2`` tag is already set.
Atomic: writes to ``<path>.tmp`` then ``os.replace``.

Example:
    uv run scripts/repack_labels_nbits.py --workers 64
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import rasterio
from rich.logging import RichHandler

log = logging.getLogger("ftw_planet.repack_labels")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--country", default="all", help="Country slug or 'all'.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _iter_labels(planet_root: Path, country: str) -> list[Path]:
    dirs = (
        [planet_root / country]
        if country != "all"
        else [d for d in planet_root.iterdir() if d.is_dir() and d.name != "_global"]
    )
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        out.extend(sorted(d.glob("*_label.tif")))
    return out


def _is_nbits2(path: Path) -> bool:
    """Cheap check: read GDAL's IMAGE_STRUCTURE tag for NBITS."""
    try:
        with rasterio.open(path) as src:
            tags = src.tags(ns="IMAGE_STRUCTURE")
            return tags.get("NBITS") == "2"
    except Exception:
        return False


def _repack_one(path: Path, dry_run: bool) -> dict:
    t = time.perf_counter()
    size_before = path.stat().st_size
    result: dict[str, Any] = {"path": str(path), "size_before": size_before}
    try:
        if _is_nbits2(path):
            result["status"] = "skipped_already_nbits2"
            return result

        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            user_tags = src.tags()  # preserve metadata tags
            desc = src.descriptions[0] if src.descriptions else None
            height = src.height
        # Rewrite with NBITS=2 + ZSTD-22, single strip.
        profile.update(
            {
                "driver": "GTiff",
                "dtype": "uint8",
                "nbits": 2,
                "compress": "ZSTD",
                "zstd_level": 22,
                "tiled": False,
                "blockysize": height,
                "predictor": 1,
            }
        )
        # Drop tiling-specific keys that conflict with stripped layout.
        profile.pop("blockxsize", None)

        tmp = path.with_suffix(path.suffix + ".tmp")
        if dry_run:
            result["status"] = "dry_run"
            return result
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data, 1)
            if user_tags:
                dst.update_tags(**user_tags)
            if desc:
                dst.set_band_description(1, desc)
        os.replace(tmp, path)
        size_after = path.stat().st_size
        result["status"] = "repacked"
        result["size_after"] = size_after
        result["ratio"] = round(size_after / max(size_before, 1), 4)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    result["wall_s"] = round(time.perf_counter() - t, 3)
    return result


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_time=True)]
    )
    for noisy in ("rasterio",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()

    todo = _iter_labels(args.planet_root, args.country)
    log.info("label tifs found: %d (country=%s)", len(todo), args.country)
    if not todo:
        return 0

    log_path = args.planet_root / "_global" / "repack_labels_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_before = 0
    total_after = 0
    n_repacked = 0
    n_skipped = 0
    n_failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex, log_path.open("a") as fout:
        futs = {ex.submit(_repack_one, p, args.dry_run): p for p in todo}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            fout.write(json.dumps(row) + "\n")
            total_before += row.get("size_before", 0)
            if row.get("status") == "repacked":
                n_repacked += 1
                total_after += row.get("size_after", 0)
            elif row.get("status", "").startswith("skipped"):
                n_skipped += 1
                total_after += row.get("size_before", 0)
            elif row["status"] == "failed":
                n_failed += 1
            done += 1
            if done % 5000 == 0:
                log.info("  %d / %d processed", done, len(todo))

    log.info(
        "done: %d repacked  %d skipped  %d failed",
        n_repacked,
        n_skipped,
        n_failed,
    )
    if total_before:
        log.info(
            "size: %.2f MiB -> %.2f MiB (%.1f%%)",
            total_before / (1024**2),
            total_after / (1024**2),
            (total_after / total_before) * 100,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
