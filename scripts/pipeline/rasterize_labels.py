"""Rasterize FTW field-boundary polygons onto Planet patch grids.

For every PlanetScope SR GeoTIFF in ``data/planet/<country>/<id>_<window>.tif``
that already exists, we look up the country's polygon GeoParquet
(``data/ftw_polygons/<country>.parquet``), reproject its polygons to the
patch's native UTM, filter to those intersecting the patch bounds, and
rasterize a 3-class label map (0=background, 1=field interior, 2=field
boundary) onto the patch's grid. Output: ``<id>_<window>_label.tif``.

The boundary class is produced by buffering each polygon's exterior ring
by ``--boundary-buffer-m`` meters (default 3 m → ~2 px at 3 m resolution)
and rasterizing on top of the interior layer.

Per-country sharding: pass ``--country <name>`` to process one country
(used by SLURM array). Skips a tif if its label is already present.

Example:
    uv run scripts/rasterize_labels.py --country rwanda
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rich.logging import RichHandler
from shapely.geometry import box

log = logging.getLogger("ftw_planet.rasterize")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", required=True, help="FTW country slug, or 'all'.")
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--polygons-root", type=Path, default=Path("data/ftw_polygons"))
    p.add_argument("--workers", type=int, default=8, help="Per-country worker pool size.")
    p.add_argument(
        "--boundary-buffer-m",
        type=float,
        default=3.0,
        help="Buffer applied to polygon exterior to make boundary class (meters).",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="If set, write a JSONL summary (one row per processed tif).",
    )
    return p.parse_args()


def _rasterize_one(
    sr_path: Path,
    polys_utm: gpd.GeoDataFrame,
    boundary_buffer_m: float,
) -> dict:
    out_path = sr_path.with_name(sr_path.stem + "_label.tif")
    if out_path.exists():
        return {"path": str(sr_path), "status": "skipped_existing"}

    try:
        with rasterio.open(sr_path) as src:
            transform = src.transform
            width, height = src.width, src.height
            crs = src.crs
            bounds = src.bounds
        pbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        hits = polys_utm[polys_utm.intersects(pbox)]
        if len(hits) == 0:
            mask = (
                0
                * rasterize(
                    [(pbox, 0)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                )
            ).astype("uint8")
            n_field = n_bound = 0
        else:
            interior = rasterize(
                [(g, 1) for g in hits.geometry],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
            boundary = rasterize(
                [(g.exterior.buffer(boundary_buffer_m), 1) for g in hits.geometry],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
            mask = interior.copy()
            mask[boundary == 1] = 2
            n_field = int((interior == 1).sum())
            n_bound = int((boundary == 1).sum())

        # NBITS=2 packs 4 pixels/byte before compression — labels use 3 values
        # (0/1/2) so 2 bits is plenty. Stripped layout (no tiling) since these
        # are small patches read whole at training time.
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "nbits": 2,
            "crs": crs,
            "transform": transform,
            "compress": "ZSTD",
            "zstd_level": 22,
            "tiled": False,
            "blockysize": height,
            "predictor": 1,  # categorical — horizontal diff hurts
            "nodata": 255,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask, 1)
        return {
            "path": str(out_path),
            "status": "ok",
            "n_polys": len(hits),
            "field_pixels": n_field,
            "boundary_pixels": n_bound,
            "total_pixels": height * width,
        }
    except Exception as e:
        return {"path": str(sr_path), "status": "failed", "error": str(e)}


def _process_country(
    country: str,
    planet_root: Path,
    polys_root: Path,
    workers: int,
    boundary_buffer_m: float,
) -> list[dict]:
    poly_path = polys_root / f"{country}.parquet"
    planet_dir = planet_root / country
    if not poly_path.exists():
        log.warning("polygons missing for %s — skip", country)
        return []
    if not planet_dir.exists():
        log.warning("planet dir missing for %s — skip", country)
        return []

    poly_df = gpd.read_parquet(poly_path)
    log.info("country=%s polys=%d", country, len(poly_df))

    # Group SR tifs by their UTM CRS so we reproject polygons once per zone.
    tifs = sorted(p for p in planet_dir.glob("*_a.tif") if "_udm2" not in p.name)
    tifs += sorted(p for p in planet_dir.glob("*_b.tif") if "_udm2" not in p.name)
    if not tifs:
        log.info("no SR tifs for %s", country)
        return []

    # Bucket tifs by CRS.
    by_crs: dict[str, list[Path]] = {}
    for tif in tifs:
        with rasterio.open(tif) as src:
            crs_str = src.crs.to_string()
        by_crs.setdefault(crs_str, []).append(tif)
    log.info("country=%s CRS zones=%d tifs=%d", country, len(by_crs), len(tifs))

    rows: list[dict] = []
    for crs_str, tif_list in by_crs.items():
        polys_utm = poly_df.to_crs(crs_str)
        log.info("  zone %s: %d tifs", crs_str, len(tif_list))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_rasterize_one, t, polys_utm, boundary_buffer_m): t for t in tif_list}
            done = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 500 == 0:
                    log.info("    %d/%d tifs done in zone", done, len(tif_list))
    return rows


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    args = parse_args()

    countries: list[str]
    if args.country == "all":
        countries = sorted(p.stem for p in args.polygons_root.glob("*.parquet"))
    else:
        countries = [args.country]

    all_rows: list[dict] = []
    for c in countries:
        all_rows.extend(
            _process_country(
                c, args.planet_root, args.polygons_root, args.workers, args.boundary_buffer_m
            )
        )

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("a") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")

    from collections import Counter

    log.info("statuses: %s", dict(Counter(r["status"] for r in all_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
