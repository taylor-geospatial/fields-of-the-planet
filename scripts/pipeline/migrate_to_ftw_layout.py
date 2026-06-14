"""Migrate flat per-country Planet patches to FTW-aligned subdir layout.

Before:
    data/planet/<country>/<id>_a.tif
    data/planet/<country>/<id>_b.tif
    data/planet/<country>/<id>_label.tif

After:
    data/planet/<country>/window_a/<id>.tif
    data/planet/<country>/window_b/<id>.tif
    data/planet/<country>/labels/<id>.tif

Idempotent: skips files already in the new layout. Dry-run by default;
pass ``--apply`` to actually move. Single-process — file moves are
I/O-bound and ~133k moves doesn't justify parallelism.

Example:
    uv run scripts/migrate_to_ftw_layout.py            # dry-run
    uv run scripts/migrate_to_ftw_layout.py --apply
"""

import argparse
import logging
import os
from pathlib import Path

from rich.logging import RichHandler

log = logging.getLogger("ftw_planet.migrate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--apply", action="store_true", help="Actually move files (default: dry-run).")
    return p.parse_args()


def _classify(name: str) -> tuple[str, str] | None:
    """Return (subdir, new_name) for a flat-layout filename, else None.

    Skips ``_udm2`` partner tifs (we leave them in the country root for now —
    they're an internal artifact, not part of the published layout).
    """
    if not name.endswith(".tif"):
        return None
    if "_udm2" in name:
        return None
    stem = name[: -len(".tif")]
    if stem.endswith("_a"):
        return ("window_a", stem[:-2] + ".tif")
    if stem.endswith("_b"):
        return ("window_b", stem[:-2] + ".tif")
    if stem.endswith("_label"):
        return ("labels", stem[: -len("_label")] + ".tif")
    return None


def _migrate_country(country_dir: Path, apply: bool) -> dict[str, int]:
    counts = {"window_a": 0, "window_b": 0, "labels": 0, "skipped": 0, "conflict": 0}
    # Only look at files directly in the country dir, not in already-migrated subdirs.
    for entry in sorted(country_dir.iterdir()):
        if not entry.is_file():
            continue
        cls = _classify(entry.name)
        if cls is None:
            counts["skipped"] += 1
            continue
        subdir, new_name = cls
        dst_dir = country_dir / subdir
        dst = dst_dir / new_name
        if dst.exists():
            log.warning("conflict %s -> %s already exists", entry, dst)
            counts["conflict"] += 1
            continue
        if apply:
            dst_dir.mkdir(parents=True, exist_ok=True)
            os.replace(entry, dst)
        counts[subdir] += 1
    return counts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    args = parse_args()
    if not args.apply:
        log.info("DRY RUN — pass --apply to actually move files")

    total = {"window_a": 0, "window_b": 0, "labels": 0, "skipped": 0, "conflict": 0}
    countries = sorted(d for d in args.planet_root.iterdir() if d.is_dir() and d.name != "_global")
    for cdir in countries:
        c = _migrate_country(cdir, args.apply)
        log.info(
            "%s: window_a=%d window_b=%d labels=%d conflict=%d skipped=%d",
            cdir.name,
            c["window_a"],
            c["window_b"],
            c["labels"],
            c["conflict"],
            c["skipped"],
        )
        for k, v in c.items():
            total[k] += v

    log.info(
        "TOTAL: window_a=%d window_b=%d labels=%d conflict=%d skipped=%d",
        total["window_a"],
        total["window_b"],
        total["labels"],
        total["conflict"],
        total["skipped"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
