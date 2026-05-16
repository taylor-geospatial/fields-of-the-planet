"""Build per-country ``index.parquet`` for the migrated FTW-aligned layout.

For each country directory under ``data/planet/`` (skipping ``_global``),
walks the migrated tifs (``window_a/``, ``window_b/``, ``labels/``), reads
their TIFF tags for provenance + UDM2 stats, joins with
``_global/manifest.jsonl`` for the patch's WGS84 geometry, and writes
a geoparquet ``<country>/index.parquet`` with one row per patch.

Patch is the unit: a row exists only when both windows AND the label
tif are present. Patches missing any component are logged + skipped.

Example:
    uv run scripts/build_country_indexes.py --workers 16
    uv run scripts/build_country_indexes.py --country rwanda
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
from rich.logging import RichHandler
from shapely.geometry import shape

log = logging.getLogger("ftw_planet.index")


# Columns in canonical order — kept here so the schema is one place.
COLUMNS: list[str] = [
    "patch_id",
    "country",
    "geometry",
    "crs",
    "bounds_4326",
    "image_a_path",
    "image_b_path",
    "label_path",
    "item_id_a",
    "item_id_b",
    "scene_date_a",
    "scene_date_b",
    "cloud_cover_a",
    "cloud_cover_b",
    "coverage_a",
    "coverage_b",
    "source_a",
    "source_b",
    "udm2_clear_a",
    "udm2_cloud_a",
    "udm2_shadow_a",
    "udm2_light_haze_a",
    "udm2_heavy_haze_a",
    "udm2_snow_a",
    "udm2_unusable_a",
    "udm2_confidence_mean_a",
    "udm2_usable_flag_a",
    "udm2_clear_b",
    "udm2_cloud_b",
    "udm2_shadow_b",
    "udm2_light_haze_b",
    "udm2_heavy_haze_b",
    "udm2_snow_b",
    "udm2_unusable_b",
    "udm2_confidence_mean_b",
    "udm2_usable_flag_b",
    "ftw_target_date_a",
    "ftw_target_date_b",
    "ftw_season_start",
    "ftw_season_end",
    "usable_pair",
]

UDM2_FLOAT_KEYS = [
    "UDM2_CLEAR",
    "UDM2_CLOUD",
    "UDM2_SHADOW",
    "UDM2_LIGHT_HAZE",
    "UDM2_HEAVY_HAZE",
    "UDM2_SNOW",
    "UDM2_UNUSABLE",
    "UDM2_CONFIDENCE_MEAN",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--country", default="all", help="One country slug or 'all'.")
    return p.parse_args()


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> bool | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _to_dt(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def _to_date(v: Any) -> date | None:
    if not v:
        return None
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        dt = _to_dt(v)
        return dt.date() if dt else None


def _load_manifest_geoms(planet_root: Path) -> dict[tuple[str, str], dict]:
    """Map (country, patch_id) -> GeoJSON geometry (window-invariant)."""
    out: dict[tuple[str, str], dict] = {}
    p = planet_root / "_global" / "manifest.jsonl"
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["country"], r["id"])
            # geometry is identical across windows for a given patch
            if key not in out:
                out[key] = r["geometry_4326"]
    return out


def _read_window_tags(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        tags = src.tags()
        crs = src.crs.to_string() if src.crs else None
        b = src.bounds
        bounds = (b.left, b.bottom, b.right, b.top)
    tags["__crs__"] = crs
    tags["__bounds__"] = bounds
    return tags


def _build_row(
    country: str,
    patch_id: str,
    img_a: Path,
    img_b: Path,
    lbl: Path,
    geom_4326: dict,
    planet_root: Path,
) -> dict[str, Any] | None:
    try:
        ta = _read_window_tags(img_a)
        tb = _read_window_tags(img_b)
    except Exception as e:
        log.warning("read fail %s/%s: %s", country, patch_id, e)
        return None

    crs = ta.get("__crs__") or tb.get("__crs__")
    bounds = ta.get("__bounds__")

    row: dict[str, Any] = {
        "patch_id": patch_id,
        "country": country,
        "geometry": shape(geom_4326),
        "crs": crs,
        "bounds_4326": _patch_bounds_4326(geom_4326),
        "image_a_path": str(img_a.relative_to(planet_root)),
        "image_b_path": str(img_b.relative_to(planet_root)),
        "label_path": str(lbl.relative_to(planet_root)),
        "item_id_a": ta.get("ITEM_ID"),
        "item_id_b": tb.get("ITEM_ID"),
        "scene_date_a": _to_dt(ta.get("SCENE_DATE")),
        "scene_date_b": _to_dt(tb.get("SCENE_DATE")),
        "cloud_cover_a": _to_float(ta.get("CLOUD_COVER")),
        "cloud_cover_b": _to_float(tb.get("CLOUD_COVER")),
        "coverage_a": _to_float(ta.get("COVERAGE")),
        "coverage_b": _to_float(tb.get("COVERAGE")),
        "source_a": ta.get("SOURCE"),
        "source_b": tb.get("SOURCE"),
        "ftw_target_date_a": _to_date(ta.get("FTW_TARGET_DATE")),
        "ftw_target_date_b": _to_date(tb.get("FTW_TARGET_DATE")),
        "ftw_season_start": _to_date(ta.get("FTW_SEASON_START") or tb.get("FTW_SEASON_START")),
        "ftw_season_end": _to_date(ta.get("FTW_SEASON_END") or tb.get("FTW_SEASON_END")),
    }
    # bounds: prefer reading from tif (it's reprojected from geom in any case;
    # but the row schema asks for 4326 bounds derived from manifest geometry).
    _ = bounds

    for suffix, t in (("a", ta), ("b", tb)):
        for k in UDM2_FLOAT_KEYS:
            col = f"{k.lower()}_{suffix}"
            row[col] = _to_float(t.get(k))
        row[f"udm2_usable_flag_{suffix}"] = _to_bool(t.get("UDM2_USABLE_FLAG"))

    ua = row.get("udm2_usable_flag_a")
    ub = row.get("udm2_usable_flag_b")
    row["usable_pair"] = bool(ua) and bool(ub)
    return row


def _patch_bounds_4326(geom_4326: dict) -> list[float]:
    g = shape(geom_4326)
    minx, miny, maxx, maxy = g.bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _process_country(country: str, planet_root: Path) -> tuple[str, int, int]:
    cdir = planet_root / country
    wa_dir = cdir / "window_a"
    wb_dir = cdir / "window_b"
    lbl_dir = cdir / "labels"
    if not wa_dir.is_dir():
        log.warning("%s: no window_a dir — skip", country)
        return (country, 0, 0)

    geoms = _load_manifest_geoms(planet_root)

    rows: list[dict[str, Any]] = []
    missing = 0
    a_tifs = sorted(wa_dir.glob("*.tif"))
    for img_a in a_tifs:
        pid = img_a.stem
        img_b = wb_dir / f"{pid}.tif"
        lbl = lbl_dir / f"{pid}.tif"
        if not img_b.exists() or not lbl.exists():
            missing += 1
            continue
        geom = geoms.get((country, pid))
        if geom is None:
            missing += 1
            continue
        r = _build_row(country, pid, img_a, img_b, lbl, geom, planet_root)
        if r is None:
            missing += 1
            continue
        rows.append(r)

    if missing:
        log.warning("%s: %d patches missing component or geometry — skipped", country, missing)

    if not rows:
        log.info("%s: no rows — skipping parquet write", country)
        return (country, 0, missing)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # Enforce column order
    gdf = gdf[COLUMNS]
    out_path = cdir / "index.parquet"
    gdf.to_parquet(out_path, compression="zstd")
    log.info("%s: wrote %s (%d rows)", country, out_path, len(gdf))
    return (country, len(gdf), missing)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    for noisy in ("rasterio", "fiona", "pyproj"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()

    if args.country == "all":
        countries = sorted(
            d.name for d in args.planet_root.iterdir() if d.is_dir() and d.name != "_global"
        )
    else:
        countries = [args.country]

    total_rows = 0
    total_missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_country, c, args.planet_root): c for c in countries}
        for fut in as_completed(futs):
            country, n, miss = fut.result()
            total_rows += n
            total_missing += miss

    log.info(
        "TOTAL: %d rows across %d countries; %d skipped", total_rows, len(countries), total_missing
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
