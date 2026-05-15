"""Build a single dataset-wide ``index.parquet`` for the migrated layout.

One row per patch, covering all countries. Output:
``data/planet/index.parquet`` (geoparquet, WGS84 polygons).

Reads provenance + UDM2 stats from each tif's TIFF tags, and pulls the
patch's WGS84 polygon from ``_global/manifest.jsonl``. A row exists only
when both window tifs AND the label tif are on disk.

Paths in the index are relative to ``--planet-root`` and include the
country, e.g. ``rwanda/window_a/1592589.tif`` — load as
``planet_root / row.image_a_path``.

Example:
    uv run scripts/build_dataset_index.py --workers 32
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
    p.add_argument("--workers", type=int, default=32)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path (default: <planet-root>/index.parquet).",
    )
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
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r["country"], r["id"])
            if key not in out:
                out[key] = r["geometry_4326"]
    return out


def _read_tif_tags(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        tags = src.tags()
        tags["__crs__"] = src.crs.to_string() if src.crs else None
    return tags


def _build_one(args: tuple[str, str, Path, Path, Path, dict, Path]) -> dict[str, Any] | None:
    country, pid, img_a, img_b, lbl, geom_4326, planet_root = args
    try:
        ta = _read_tif_tags(img_a)
        tb = _read_tif_tags(img_b)
    except Exception as e:
        log.warning("read fail %s/%s: %s", country, pid, e)
        return None

    g = shape(geom_4326)
    minx, miny, maxx, maxy = g.bounds
    row: dict[str, Any] = {
        "patch_id": pid,
        "country": country,
        "geometry": g,
        "crs": ta.get("__crs__") or tb.get("__crs__"),
        "bounds_4326": [float(minx), float(miny), float(maxx), float(maxy)],
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
    for suffix, t in (("a", ta), ("b", tb)):
        for k in UDM2_FLOAT_KEYS:
            row[f"{k.lower()}_{suffix}"] = _to_float(t.get(k))
        row[f"udm2_usable_flag_{suffix}"] = _to_bool(t.get("UDM2_USABLE_FLAG"))
    row["usable_pair"] = bool(row["udm2_usable_flag_a"]) and bool(row["udm2_usable_flag_b"])
    return row


def _collect_tasks(planet_root: Path, geoms: dict) -> tuple[list, int]:
    tasks = []
    missing = 0
    countries = sorted(
        d for d in planet_root.iterdir() if d.is_dir() and d.name != "_global"
    )
    for cdir in countries:
        wa_dir = cdir / "window_a"
        wb_dir = cdir / "window_b"
        lbl_dir = cdir / "labels"
        if not wa_dir.is_dir():
            continue
        for img_a in sorted(wa_dir.glob("*.tif")):
            pid = img_a.stem
            img_b = wb_dir / f"{pid}.tif"
            lbl = lbl_dir / f"{pid}.tif"
            if not img_b.exists() or not lbl.exists():
                missing += 1
                continue
            geom = geoms.get((cdir.name, pid))
            if geom is None:
                missing += 1
                continue
            tasks.append((cdir.name, pid, img_a, img_b, lbl, geom, planet_root))
    return tasks, missing


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_time=True)]
    )
    for noisy in ("rasterio", "fiona", "pyproj"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()

    out_path = args.out or (args.planet_root / "index.parquet")

    log.info("loading manifest geometries...")
    geoms = _load_manifest_geoms(args.planet_root)
    log.info("manifest geometries: %d", len(geoms))

    log.info("scanning patch trees...")
    tasks, missing = _collect_tasks(args.planet_root, geoms)
    log.info("candidate patches: %d (skipped %d for missing component/geom)", len(tasks), missing)
    if not tasks:
        log.warning("no patches found — wrote nothing")
        return 0

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_build_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
            done += 1
            if done % 5000 == 0:
                log.info("  %d / %d processed", done, len(tasks))

    log.info("building GeoDataFrame (%d rows)...", len(rows))
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[COLUMNS]

    # Spatially sort by Hilbert curve so row groups are geographically
    # clustered. Without this, every row group's bbox spans the planet and
    # bbox-pruning (the whole point of GeoParquet 1.1 covering) does nothing.
    log.info("computing Hilbert order for spatial row-group clustering...")
    gdf = gdf.iloc[gdf.geometry.hilbert_distance().argsort()].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s ...", out_path)
    # GeoParquet 1.1.0 + bbox covering struct -> DuckDB/duckdb-wasm spatial
    # readers can prune row groups by bbox without parsing WKB. Row group
    # size is tuned so each group covers a small geographic region after
    # Hilbert sorting (~13 groups across the dataset).
    gdf.to_parquet(
        out_path,
        compression="zstd",
        schema_version="1.1.0",
        write_covering_bbox=True,
        row_group_size=5000,
    )

    by_country = gdf.groupby("country").size()
    log.info("done: %d rows across %d countries", len(gdf), by_country.size)
    for c, n in by_country.items():
        log.info("  %-14s %d", c, int(n))
    n_usable = int(gdf["usable_pair"].sum())
    log.info("usable_pair=True: %d / %d (%.1f%%)", n_usable, len(gdf), 100 * n_usable / len(gdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
