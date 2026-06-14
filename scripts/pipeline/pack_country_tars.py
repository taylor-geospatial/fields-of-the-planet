"""Pack per-country WebDataset-compatible tar shards.

For each country, writes ``data/planet/bulk/<country>.tar`` containing
WebDataset-style samples grouped by ``patch_id``. Per patch:

    <pid>.window_a.tif    # PlanetScope SR, window A
    <pid>.window_b.tif    # PlanetScope SR, window B
    <pid>.label.tif       # 3-class semantic label (NBITS=2)
    <pid>.json            # per-patch metadata row from index.parquet

The tar is stored (not gzipped) — TIFFs are already ZSTD-22 compressed,
gzip on top just burns CPU. Webdataset loaders treat each tar as a shard
and stream samples sequentially; the per-sample json carries scene
dates, UDM2 stats, geometry, etc., so trainers don't need a sidecar.

The same tar is also a fine bulk-download artifact for non-WDS users:
``tar -xf rwanda.tar`` recovers the per-patch files.

Reads the dataset-wide ``index.parquet`` (one row per patch) and tars
only patches that exist in it.

Example:
    uv run scripts/pack_country_tars.py --workers 8
    uv run scripts/pack_country_tars.py --country rwanda
"""

import argparse
import io
import json
import logging
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
from rich.logging import RichHandler
from shapely.geometry import mapping

log = logging.getLogger("ftw_planet.pack_tars")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to index.parquet (default: <root>/index.parquet).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/bulk).",
    )
    p.add_argument("--country", default="all", help="One country slug or 'all'.")
    p.add_argument("--workers", type=int, default=4, help="Countries packed in parallel.")
    return p.parse_args()


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # numpy types arrive after parquet round-trip even though we wrote
    # Python lists — geopandas/pyarrow promotes them. Handle scalars + arrays.
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"unserialisable type {type(o)}")


def _row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Strip path columns (recoverable from sample key) and embed geometry as GeoJSON."""
    out = {
        k: v
        for k, v in row.items()
        if k not in ("image_a_path", "image_b_path", "label_path", "geometry")
    }
    geom = row.get("geometry")
    if geom is not None:
        out["geometry"] = mapping(geom)
    return out


def _add_file(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    info = tar.gettarinfo(name=str(src), arcname=arcname)
    # Normalise — uid/gid/mtime irrelevant for a published artifact.
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    with src.open("rb") as f:
        tar.addfile(info, f)


def _add_bytes(tar: tarfile.TarFile, data: bytes, arcname: str, mtime: float) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = int(mtime)
    tar.addfile(info, io.BytesIO(data))


def _pack_country(
    country: str, gdf: gpd.GeoDataFrame, planet_root: Path, out_dir: Path
) -> tuple[str, int, int]:
    sub = gdf[gdf["country"] == country].sort_values("patch_id")
    if len(sub) == 0:
        log.warning("%s: no rows in index — skip", country)
        return (country, 0, 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{country}.tar"
    tmp_path = out_path.with_suffix(".tar.tmp")

    n_ok = 0
    n_missing = 0
    # mtime constant so the tar is deterministic enough across re-packs.
    mtime = datetime(2026, 1, 1).timestamp()
    with tarfile.open(tmp_path, "w") as tar:
        for row in sub.to_dict(orient="records"):
            pid = row["patch_id"]
            img_a = planet_root / row["image_a_path"]
            img_b = planet_root / row["image_b_path"]
            lbl = planet_root / row["label_path"]
            if not (img_a.exists() and img_b.exists() and lbl.exists()):
                n_missing += 1
                continue
            try:
                # Encode metadata first — if json fails we haven't yet added
                # any tifs to the tar, so the shard stays consistent.
                meta_bytes = json.dumps(_row_to_metadata(row), default=_json_default).encode()
                _add_file(tar, img_a, f"{pid}.window_a.tif")
                _add_file(tar, img_b, f"{pid}.window_b.tif")
                _add_file(tar, lbl, f"{pid}.label.tif")
                _add_bytes(tar, meta_bytes, f"{pid}.json", mtime)
                n_ok += 1
            except Exception as e:
                log.warning("%s/%s: tar fail %s", country, pid, e)
                n_missing += 1

    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / (1024**2)
    log.info(
        "%s: wrote %s (%d samples, %.1f MiB, %d missing)",
        country,
        out_path,
        n_ok,
        size_mb,
        n_missing,
    )
    return (country, n_ok, n_missing)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_time=True)]
    )
    for noisy in ("fiona", "pyproj"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()

    index_path = args.index or (args.planet_root / "index.parquet")
    out_dir = args.out_dir or (args.planet_root / "bulk")

    log.info("reading %s ...", index_path)
    gdf = gpd.read_parquet(index_path)
    log.info("index rows: %d", len(gdf))

    if args.country == "all":
        countries = sorted(gdf["country"].unique().tolist())
    else:
        countries = [args.country]
    log.info("countries: %d", len(countries))

    total_ok = 0
    total_missing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_pack_country, c, gdf, args.planet_root, out_dir): c for c in countries}
        for fut in as_completed(futs):
            _, n_ok, n_missing = fut.result()
            total_ok += n_ok
            total_missing += n_missing

    log.info("TOTAL: %d samples packed, %d skipped (missing files)", total_ok, total_missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
