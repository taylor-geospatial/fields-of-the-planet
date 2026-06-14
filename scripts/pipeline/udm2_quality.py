"""Compute UDM2 quality stats per Planet patch.

Reads every ``<id>_<window>_udm2.tif`` under ``data/planet/<country>/``,
computes per-band coverage percentages (clear / snow / shadow / haze /
cloud / unusable), and writes a single JSONL file with one row per UDM2.

UDM2 band semantics (Planet Surface Reflectance products v2):
    1 = clear
    2 = snow
    3 = shadow
    4 = light haze
    5 = heavy haze
    6 = cloud
    7 = confidence (%)
    8 = unusable_data (catch-all bitfield)

Output rows include a ``flag`` field summarising whether the patch is
likely usable for training (``clear>=0.95`` and ``unusable<=0.05``).

Example:
    uv run scripts/udm2_quality.py --out data/planet/_global/udm2_quality.jsonl
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rich.logging import RichHandler

log = logging.getLogger("ftw_planet.udm2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--country", default="all", help="country slug or 'all'")
    p.add_argument("--out", type=Path, default=Path("data/planet/_global/udm2_quality.jsonl"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--clear-threshold",
        type=float,
        default=0.95,
        help="Min fraction of clear pixels to flag a patch as usable.",
    )
    return p.parse_args()


def _stat_one(tif_path: Path, clear_threshold: float) -> dict:
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()
            n_pix = data.shape[1] * data.shape[2]
            clear = float((data[0] > 0).sum()) / n_pix
            snow = float((data[1] > 0).sum()) / n_pix
            shadow = float((data[2] > 0).sum()) / n_pix
            l_haze = float((data[3] > 0).sum()) / n_pix
            h_haze = float((data[4] > 0).sum()) / n_pix
            cloud = float((data[5] > 0).sum()) / n_pix
            conf = float(data[6].mean())
            unusable = float((data[7] > 0).sum()) / n_pix
        # Parse country/id/window from path
        country = tif_path.parent.name
        # name format: <id>_<a|b>_udm2.tif — strip suffix
        stem = tif_path.stem  # e.g. 1592589_a_udm2
        parts = stem.rsplit("_", 2)  # <id>, <window>, 'udm2'
        if len(parts) == 3 and parts[2] == "udm2":
            pid = parts[0]
            window = parts[1]
        else:
            pid = stem
            window = ""
        usable = clear >= clear_threshold and unusable <= (1 - clear_threshold)
        return {
            "country": country,
            "id": pid,
            "window": window,
            "clear": round(clear, 4),
            "snow": round(snow, 4),
            "shadow": round(shadow, 4),
            "light_haze": round(l_haze, 4),
            "heavy_haze": round(h_haze, 4),
            "cloud": round(cloud, 4),
            "unusable": round(unusable, 4),
            "confidence": round(conf, 1),
            "usable_flag": bool(usable),
        }
    except Exception as e:
        return {"path": str(tif_path), "status": "failed", "error": str(e)}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    args = parse_args()

    if args.country == "all":
        country_dirs = [d for d in args.planet_root.iterdir() if d.is_dir() and d.name != "_global"]
    else:
        country_dirs = [args.planet_root / args.country]

    all_tifs: list[Path] = []
    for d in country_dirs:
        all_tifs.extend(sorted(d.glob("*_udm2.tif")))
    log.info("found %d UDM2 tifs across %d countries", len(all_tifs), len(country_dirs))
    if not all_tifs:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # Overwrite mode: this script computes a *current* snapshot per UDM2 tif on
    # disk. Appending across runs creates stale duplicates; we don't want that.
    with ProcessPoolExecutor(max_workers=args.workers) as ex, args.out.open("w") as fout:
        futs = {ex.submit(_stat_one, t, args.clear_threshold): t for t in all_tifs}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            fout.write(json.dumps(row) + "\n")
            rows.append(row)
            done += 1
            if done % 5000 == 0:
                log.info("  %d / %d done", done, len(all_tifs))

    # Quick summary
    ok = [r for r in rows if "usable_flag" in r]
    usable = sum(1 for r in ok if r["usable_flag"])
    log.info(
        "processed %d, usable=%d (%.1f%%), failed=%d",
        len(ok),
        usable,
        100 * usable / max(len(ok), 1),
        len(rows) - len(ok),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
