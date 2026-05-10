"""Planet Data API + COG range-read helpers for matching FTW patches.

Strategy
--------
For each FTW patch and each window (a/b):
  1. Search PSScene (Data API) within +/-search_days of the FTW S2 date,
     intersecting the patch polygon, with cloud_cover <= max_cloud_cover.
  2. Rank candidates by (full-coverage first, then |Δdate|, then cloud_cover)
     and pick the best.
  3. Activate the SR + UDM2 assets, fetch the signed Google Cloud Storage
     URLs, then range-read just the patch window via GDAL/`/vsicurl/` and
     reproject to ``target_crs`` at ``resolution_m`` resolution.

Why range reads instead of the Orders API: PSScene `ortho_analytic_4b_sr`
is a Cloud-Optimized GeoTIFF. The signed location URL is fetched once;
GDAL pulls only the COG header + the tiles overlapping the patch (typically
~1-2 MB out of a ~500 MB scene strip). No server-side queue, no full-strip
download. Activation latency still applies on cold scenes (~seconds-minutes)
but subsequent reads are immediate.
"""

import asyncio
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import rasterio
import rasterio.windows
from planet import Session, data_filter
from planet.exceptions import APIError, TooManyRequests
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

log = logging.getLogger(__name__)

ITEM_TYPE = "PSScene"
ASSET_SR = "ortho_analytic_4b_sr"  # 4-band surface reflectance COG
ASSET_UDM2 = "ortho_udm2"  # usable-data mask v2

# GDAL knobs for fast HTTP-range COG reads. Apply via rasterio.Env.
# Numeric ones like GDAL_CACHEMAX go through C funcs that require int — keep
# them as int (rasterio passes int kwargs through correctly; strings break).
GDAL_COG_OPTS: dict[str, Any] = {
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # No CPL_VSIL_CURL_ALLOWED_EXTENSIONS filter — Planet's signed URLs end
    # with `?token=...`, no .tif suffix, so an extension allowlist breaks them.
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "CPL_VSIL_CURL_CACHE_SIZE": 200_000_000,
    "GDAL_INGESTED_BYTES_AT_OPEN": 32_768,
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": 67_108_864,
    "GDAL_CACHEMAX": 1024,  # MB
    "GDAL_HTTP_CONNECTTIMEOUT": 10,
    "GDAL_HTTP_TIMEOUT": 60,
    "GDAL_HTTP_MAX_RETRY": 5,
    "GDAL_HTTP_RETRY_DELAY": 2,
}


@dataclass
class CandidateScene:
    item_id: str
    acquired: datetime
    cloud_cover: float
    coverage: float


def _build_filter(
    geometry_geojson: dict,
    target_date: datetime,
    search_days: int,
    max_cloud_cover: float,
) -> dict:
    gte = (target_date - timedelta(days=search_days)).replace(tzinfo=UTC)
    lte = (target_date + timedelta(days=search_days)).replace(tzinfo=UTC)
    return data_filter.and_filter(
        [
            data_filter.geometry_filter(geometry_geojson),
            data_filter.date_range_filter("acquired", gte=gte, lte=lte),
            data_filter.range_filter("cloud_cover", lte=max_cloud_cover),
            data_filter.permission_filter(),
            data_filter.std_quality_filter(),
        ]
    )


async def _with_retries(awaitable_fn: Any, *args: Any, max_attempts: int = 8, **kw: Any) -> Any:
    """Retry on 429 + 5xx with exponential backoff. Re-awaits the factory each attempt."""
    delay = 0.5
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await awaitable_fn(*args, **kw)
        except TooManyRequests as e:
            last = e
            await asyncio.sleep(delay + (attempt * 0.1))
            delay = min(delay * 2, 30.0)
        except (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
        ) as e:
            if attempt < max_attempts - 1:
                last = e
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise
        except APIError as e:
            # planet's APIError lumps 4xx + 5xx; detect server errors heuristically
            # since the SDK doesn't expose a clean status_code attribute on all paths.
            msg = str(e)
            is_server = (
                "503" in msg
                or "502" in msg
                or "504" in msg
                or "Server Error" in msg
                or "service" in msg.lower()
            )
            if is_server and attempt < max_attempts - 1:
                last = e
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise
    raise last or RuntimeError("retry loop exhausted")


async def search_best_scene(
    sess: Session,
    geometry_geojson: dict,
    target_date: datetime,
    search_days: int,
    max_cloud_cover: float,
    min_coverage: float = 0.99,
) -> CandidateScene | None:
    cl: Any = sess.client("data")  # SDK return type is unhelpful; treat as Any
    sf = _build_filter(geometry_geojson, target_date, search_days, max_cloud_cover)

    patch_geom = shape(geometry_geojson)
    patch_area = max(patch_geom.area, 1e-12)

    async def _collect() -> list[dict]:
        out: list[dict] = []
        async for it in cl.search([ITEM_TYPE], search_filter=sf, limit=200):
            out.append(it)
        return out

    items = await _with_retries(_collect)

    best: CandidateScene | None = None
    for item in items:
        props = item["properties"]
        try:
            scene_geom = shape(item["geometry"])
        except (KeyError, ValueError):
            continue
        coverage = patch_geom.intersection(scene_geom).area / patch_area
        if coverage < min_coverage:
            continue
        cand = CandidateScene(
            item_id=item["id"],
            acquired=datetime.fromisoformat(props["acquired"]),
            cloud_cover=float(props.get("cloud_cover", 1.0)),
            coverage=coverage,
        )
        if best is None or _is_better(cand, best, target_date):
            best = cand
    return best


def _is_better(a: CandidateScene, b: CandidateScene, target: datetime) -> bool:
    target_utc = target.replace(tzinfo=UTC) if target.tzinfo is None else target
    da = abs((a.acquired - target_utc).total_seconds())
    db = abs((b.acquired - target_utc).total_seconds())
    if da != db:
        return da < db
    return a.cloud_cover < b.cloud_cover


async def activate_asset_url(sess: Session, item_id: str, asset_type: str) -> str:
    """Activate a PSScene asset and return its signed GCS download URL."""
    cl: Any = sess.client("data")  # SDK return type is unhelpful; treat as Any
    assets = await _with_retries(cl.list_item_assets, ITEM_TYPE, item_id)
    asset = assets.get(asset_type)
    if asset is None:
        raise RuntimeError(f"asset {asset_type} not present on item {item_id}")
    if asset.get("status") != "active":
        await _with_retries(cl.activate_asset, asset)
        asset = await _with_retries(cl.wait_asset, asset)
    location = asset.get("location")
    if not location:
        raise RuntimeError(f"asset {asset_type} for {item_id} has no location after activation")
    return location


@contextmanager
def cog_env() -> Any:
    """rasterio.Env with the COG range-read tuning applied."""
    with rasterio.Env(**GDAL_COG_OPTS):
        yield


def aoi_bounds_in_target(aoi_4326: dict, target_crs: str) -> tuple[float, float, float, float]:
    geom = shape(aoi_4326)
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    proj_geom = shp_transform(transformer.transform, geom)
    return proj_geom.bounds  # (minx, miny, maxx, maxy)


def require_api_key() -> str:
    key = os.environ.get("PL_API_KEY")
    if not key:
        raise RuntimeError(
            "PL_API_KEY not set — populate .env from .env.example and load it before running."
        )
    return key


async def gather_with_concurrency(n: int, *coros: Any) -> list[Any]:
    sem = asyncio.Semaphore(n)

    async def _wrap(c: Any) -> Any:
        async with sem:
            return await c

    return await asyncio.gather(*(_wrap(c) for c in coros), return_exceptions=True)
