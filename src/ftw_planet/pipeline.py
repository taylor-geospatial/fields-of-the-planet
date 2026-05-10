"""Three-phase FTW->PlanetScope match pipeline.

Phases (each with its own concurrency knob, each writing a resumable JSONL cache):

  1. ``search_all``       — for every patch+window, search Data API for the
                             best PSScene candidate. Cache: ``search.jsonl``.
  2. ``activate_all``     — dedup unique item_ids from phase 1, activate each
                             scene's SR + UDM2 assets once. Cache:
                             ``activations.jsonl`` (item_id -> SR/UDM2 URLs).
  3. ``extract_all``      — for every (patch, window) row from phase 1, look
                             up the warm URLs from phase 2, range-read the
                             window, and write the local GeoTIFF. Cache:
                             ``extracts.jsonl`` (with split network_read /
                             disk_write timings).

Each phase is idempotent / resumable: rows already in the JSONL cache are
skipped on rerun. So you can re-run the whole script after a flake without
re-doing finished work.
"""

import asyncio
import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import rasterio
import rasterio.windows
from planet import Session

from ftw_planet.planet import (
    ASSET_SR,
    ASSET_UDM2,
    activate_asset_url,
    aoi_bounds_in_target,
    cog_env,
    search_best_scene,
)

log = logging.getLogger(__name__)

WINDOWS = ("a", "b")


# ---------------------------------------------------------------------------
# Resumable JSONL cache
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader — skips torn lines (rare NFS-append race)."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Phase 1 — search
# ---------------------------------------------------------------------------


async def _search_one(
    sess: Session,
    patch: dict,
    window: str,
    *,
    max_cloud_cover: float,
    min_coverage: float,
    user_search_days: int,
) -> dict:
    target_date = datetime.fromisoformat(patch[f"win_{window}_date"])
    rng = patch[f"win_{window}_range"]
    auto_days = max(
        1,
        (datetime.fromisoformat(rng[1]) - datetime.fromisoformat(rng[0])).days // 2 + 1,
    )
    search_days = user_search_days if user_search_days > 0 else auto_days
    geom = patch["geometry_4326"]

    t0 = time.perf_counter()
    best = await search_best_scene(
        sess,
        geometry_geojson=geom,
        target_date=target_date,
        search_days=search_days,
        max_cloud_cover=max_cloud_cover,
        min_coverage=min_coverage,
    )
    expanded = False
    if best is None:
        expanded = True
        best = await search_best_scene(
            sess,
            geometry_geojson=geom,
            target_date=target_date,
            search_days=search_days * 2,
            max_cloud_cover=max_cloud_cover,
            min_coverage=min_coverage,
        )
    elapsed = round(time.perf_counter() - t0, 3)

    base = {
        "id": patch["id"],
        "window": window,
        "ftw_date": patch[f"win_{window}_date"],
        "expanded_window": expanded,
        "search_s": elapsed,
    }
    if best is None:
        return {**base, "status": "no_candidate"}
    return {
        **base,
        "status": "found",
        "item_id": best.item_id,
        "scene_date": best.acquired.isoformat(),
        "cloud_cover": best.cloud_cover,
        "coverage": best.coverage,
    }


async def search_all(
    sess: Session,
    patches: list[dict],
    *,
    cache_path: Path,
    concurrency: int,
    max_cloud_cover: float,
    min_coverage: float,
    search_days: int,
) -> list[dict]:
    done = {(r["id"], r["window"]): r for r in _read_jsonl(cache_path)}
    log.info("phase1 search: %d cached, %d patches total", len(done), len(patches))

    todo: list[tuple[dict, str]] = [
        (p, w) for p in patches for w in WINDOWS if (p["id"], w) not in done
    ]
    sem = asyncio.Semaphore(concurrency)

    async def _wrapped(patch: dict, window: str) -> dict:
        async with sem:
            return await _search_one(
                sess,
                patch,
                window,
                max_cloud_cover=max_cloud_cover,
                min_coverage=min_coverage,
                user_search_days=search_days,
            )

    new_rows: list[dict] = []
    if todo:
        coros = [_wrapped(p, w) for p, w in todo]
        for fut in asyncio.as_completed(coros):
            row = await fut
            _append_jsonl(cache_path, row)
            new_rows.append(row)

    all_rows = list(done.values()) + new_rows
    log.info(
        "phase1 done: %d found, %d no_candidate",
        sum(1 for r in all_rows if r.get("status") == "found"),
        sum(1 for r in all_rows if r.get("status") == "no_candidate"),
    )
    return all_rows


# ---------------------------------------------------------------------------
# Phase 2 — activate
# ---------------------------------------------------------------------------


async def _activate_one(sess: Session, item_id: str) -> dict:
    row: dict[str, Any] = {"item_id": item_id}
    t0 = time.perf_counter()
    try:
        row["sr_url"] = await activate_asset_url(sess, item_id, ASSET_SR)
    except Exception as e:
        row["sr_url"] = None
        row["sr_error"] = str(e)
    row["activate_sr_s"] = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    try:
        row["udm2_url"] = await activate_asset_url(sess, item_id, ASSET_UDM2)
    except Exception as e:
        row["udm2_url"] = None
        row["udm2_error"] = str(e)
    row["activate_udm2_s"] = round(time.perf_counter() - t1, 3)
    return row


async def activate_all(
    sess: Session,
    item_ids: Iterable[str],
    *,
    cache_path: Path,
    concurrency: int,
) -> dict[str, dict]:
    done = {r["item_id"]: r for r in _read_jsonl(cache_path)}
    unique = sorted(set(item_ids))
    todo = [iid for iid in unique if iid not in done]
    log.info("phase2 activate: %d cached, %d to activate", len(done), len(todo))

    sem = asyncio.Semaphore(concurrency)

    async def _wrapped(iid: str) -> dict:
        async with sem:
            return await _activate_one(sess, iid)

    if todo:
        coros = [_wrapped(iid) for iid in todo]
        for fut in asyncio.as_completed(coros):
            row = await fut
            _append_jsonl(cache_path, row)
            done[row["item_id"]] = row
    return done


# ---------------------------------------------------------------------------
# Phase 3 — extract
# ---------------------------------------------------------------------------


def _read_window(
    src: "rasterio.io.DatasetReader", aoi_4326: dict
) -> tuple[Any, Any, str, int, int]:
    import math

    src_crs = src.crs.to_string()
    bounds_native = aoi_bounds_in_target(aoi_4326, src_crs)
    w = rasterio.windows.from_bounds(*bounds_native, transform=src.transform)
    col_off = math.floor(w.col_off)
    row_off = math.floor(w.row_off)
    width = math.ceil(w.col_off + w.width) - col_off
    height = math.ceil(w.row_off + w.height) - row_off
    window = rasterio.windows.Window.from_slices(
        (row_off, row_off + height), (col_off, col_off + width), boundless=True
    )
    data = src.read(window=window)
    transform = src.window_transform(window)
    return data, transform, src_crs, width, height


def _write_geotiff(
    out_path: Path,
    src: "rasterio.io.DatasetReader",
    data: Any,
    transform: Any,
    src_crs: str,
    width: int,
    height: int,
    *,
    compress: str = "ZSTD",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": src.count,
        "dtype": src.dtypes[0],
        "crs": src_crs,
        "transform": transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 2 if src.dtypes[0].startswith(("int", "uint", "float")) else 1,
        "nodata": src.nodata,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        if src.descriptions:
            dst.descriptions = src.descriptions


def _extract_scene_group(
    item_id: str,
    sr_url: str | None,
    udm2_url: str | None,
    members: list[dict],  # rows of {id, window, geometry_4326}
    out_dir: Path,
) -> list[dict]:
    """Open the scene COG once, range-read every member's window, write each.

    Members sharing one scene benefit from:
    - one COG header fetch + one TLS handshake reused across all reads
    - GDAL's per-dataset block cache amortising overlapping tile reads
    - GCS edge cache stays warm across the whole burst
    """
    out_rows: list[dict] = []
    if not sr_url:
        for m in members:
            out_rows.append(
                {"id": m["id"], "window": m["window"], "item_id": item_id, "status": "no_url"}
            )
        return out_rows

    with cog_env():
        try:
            sr_src = rasterio.open(sr_url)
        except Exception as e:
            log.warning("SR open failed for %s: %s", item_id, e)
            for m in members:
                out_rows.append(
                    {
                        "id": m["id"],
                        "window": m["window"],
                        "item_id": item_id,
                        "status": "open_failed",
                        "error": str(e),
                    }
                )
            return out_rows

        udm2_src = None
        if udm2_url:
            try:
                udm2_src = rasterio.open(udm2_url)
            except Exception as e:
                log.info("UDM2 open failed for %s: %s", item_id, e)

        try:
            for m in members:
                pid, win, aoi = m["id"], m["window"], m["geometry_4326"]
                sr_path = out_dir / f"{pid}_{win}.tif"
                udm2_path = out_dir / f"{pid}_{win}_udm2.tif"
                row: dict[str, Any] = {"id": pid, "window": win, "item_id": item_id}
                if sr_path.exists():
                    row["status"] = "skipped_existing"
                    out_rows.append(row)
                    continue
                try:
                    t0 = time.perf_counter()
                    data, transform, crs, w, h = _read_window(sr_src, aoi)
                    row["sr_read_s"] = round(time.perf_counter() - t0, 3)
                    t1 = time.perf_counter()
                    _write_geotiff(sr_path, sr_src, data, transform, crs, w, h)
                    row["sr_write_s"] = round(time.perf_counter() - t1, 3)
                    row["src_crs"] = crs

                    if udm2_src is not None:
                        t0 = time.perf_counter()
                        u_data, u_t, u_crs, u_w, u_h = _read_window(udm2_src, aoi)
                        row["udm2_read_s"] = round(time.perf_counter() - t0, 3)
                        t1 = time.perf_counter()
                        _write_geotiff(udm2_path, udm2_src, u_data, u_t, u_crs, u_w, u_h)
                        row["udm2_write_s"] = round(time.perf_counter() - t1, 3)
                        row["udm2"] = True
                    row["status"] = "matched"
                except Exception as e:
                    row["status"] = "extract_failed"
                    row["error"] = str(e)
                    log.warning("extract failed for %s/%s (%s): %s", pid, win, item_id, e)
                out_rows.append(row)
        finally:
            sr_src.close()
            if udm2_src is not None:
                udm2_src.close()
    return out_rows


async def extract_all(
    search_rows: list[dict],
    patches_by_id: dict[str, dict],
    activations: dict[str, dict],
    *,
    out_dir: Path,
    cache_path: Path,
    concurrency: int,
) -> list[dict]:
    """Group rows by scene; process each scene in its own thread (warm COG + cache)."""
    done = {(r["id"], r["window"]): r for r in _read_jsonl(cache_path)}
    todo_rows = [
        r for r in search_rows if r.get("status") == "found" and (r["id"], r["window"]) not in done
    ]

    # Group by item_id.
    groups: dict[str, list[dict]] = {}
    for r in todo_rows:
        m = {
            "id": r["id"],
            "window": r["window"],
            "geometry_4326": patches_by_id[r["id"]]["geometry_4326"],
        }
        groups.setdefault(r["item_id"], []).append(m)

    log.info(
        "phase3 extract: %d cached, %d to extract across %d scenes",
        len(done),
        len(todo_rows),
        len(groups),
    )

    sem = asyncio.Semaphore(concurrency)

    async def _wrapped_scene(item_id: str, members: list[dict]) -> list[dict]:
        async with sem:
            act = activations.get(item_id, {})
            return await asyncio.to_thread(
                _extract_scene_group,
                item_id,
                act.get("sr_url"),
                act.get("udm2_url"),
                members,
                out_dir,
            )

    if groups:
        coros = [_wrapped_scene(iid, members) for iid, members in groups.items()]
        for fut in asyncio.as_completed(coros):
            rows = await fut
            for row in rows:
                _append_jsonl(cache_path, row)
                done[(row["id"], row["window"])] = row
    return list(done.values())
