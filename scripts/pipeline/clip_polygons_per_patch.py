"""Clip FTW field-boundary polygons to per-patch bounds.

For each test patch, take the country's original FTW vector polygons
(``data/ftw_polygons/<country>.parquet``, EPSG:4326), reproject to the patch's
PlanetScope UTM grid, and clip to the patch bounds -- the same vector source and
CRS that ``scripts/pipeline/rasterize_labels.py`` rasterizes. Writes one
GeoParquet per patch (``data/ftw_polygons_clipped/<country>/<patch_id>.parquet``)
so polygon-quality metrics can score predictions against the TRUE field geometry
instead of connected components of the rasterized GT mask.

CPU/IO only -- run on the ``cpu_amd`` partition, never a GPU node.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

# Attributes worth carrying into the per-patch files (geometry added separately).
KEEP_COLS = ("id", "crop_id", "crop_name", "area", "perimeter")


def _test_patch_ids(country: str, split: str) -> list[str]:
    chips = gpd.read_parquet(f"data/ftw/{country}/chips_{country}.parquet")
    return [str(a) for a in chips.loc[chips["split"] == split, "aoi_id"]]


def clip_country(
    country: str, split: str, window: str, out_root: Path, overwrite: bool
) -> tuple[int, int]:
    poly_path = Path(f"data/ftw_polygons/{country}.parquet")
    chips_path = Path(f"data/ftw/{country}/chips_{country}.parquet")
    if not poly_path.exists() or not chips_path.exists():
        print(f"  [skip] {country}: missing polygons or chips")
        return 0, 0

    polys = gpd.read_parquet(poly_path)
    keep = [c for c in KEEP_COLS if c in polys.columns] + ["geometry"]
    polys = polys[keep]

    out_dir = out_root / country
    out_dir.mkdir(parents=True, exist_ok=True)

    # Most patches in a country share one UTM zone, so cache the reprojected
    # GeoDataFrame + its spatial index per EPSG to avoid reprojecting per patch.
    reproj_cache: dict[int, tuple[gpd.GeoDataFrame, object]] = {}
    written = skipped = 0

    for pid in _test_patch_ids(country, split):
        out = out_dir / f"{pid}.parquet"
        if out.exists() and not overwrite:
            skipped += 1
            continue
        tif = Path(f"data/planet/{country}/window_{window}/{pid}.tif")
        if not tif.exists():
            skipped += 1
            continue
        with rasterio.open(tif) as src:
            epsg = src.crs.to_epsg()
            bounds = src.bounds

        if epsg not in reproj_cache:
            g = polys.to_crs(epsg=epsg)
            reproj_cache[epsg] = (g, g.sindex)
        g_utm, sidx = reproj_cache[epsg]

        bbox = box(*bounds)
        cand = g_utm.iloc[list(sidx.query(bbox, predicate="intersects"))]
        clipped = gpd.clip(cand, bbox, keep_geom_type=True)
        clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
        # True planimetric area on the UTM grid (m^2 -> ha), for area-binned eval.
        clipped["area_ha"] = clipped.geometry.area / 1e4
        clipped.to_parquet(out)
        written += 1

    print(f"  {country}: wrote {written}, skipped {skipped} ({split})")
    return written, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--countries",
        nargs="*",
        default=None,
        help="Subset of countries; default = every country with a polygon parquet.",
    )
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--window", default="a", help="Planet window whose grid defines patch bounds.")
    p.add_argument("--out-root", default="data/ftw_polygons_clipped", type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    countries = args.countries or sorted(
        p.stem for p in Path("data/ftw_polygons").glob("*.parquet")
    )
    print(f"clipping {len(countries)} countries (split={args.split}) -> {args.out_root}")

    total_w = total_s = 0
    for country in countries:
        w, s = clip_country(country, args.split, args.window, args.out_root, args.overwrite)
        total_w += w
        total_s += s
    print(f"done. wrote {total_w} patch parquets, skipped {total_s}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
