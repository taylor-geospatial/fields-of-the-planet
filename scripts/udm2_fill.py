"""Fill UDM2 masks for PlanetScope SR tifs that lack a companion ``_udm2.tif``.

Designed to run in parallel with the SR-only resample, since UDM2
activation is ~3x slower than SR. By polling for SR-only patches and
activating UDM2 as soon as their scenes are known, we overlap UDM2
thawing with SR extraction instead of doing it serially after.

Workflow per pass:
  1. Walk ``data/planet/<country>/*_<a|b>.tif`` that have no matching
     ``*_<a|b>_udm2.tif`` partner.
  2. Resolve each (country, id, window) -> item_id by reading
     ``_global/resample_log.jsonl`` and ``_global/extract/shard_*.jsonl``
     (last-write-wins).
  3. Group by item_id, bulk-activate UDM2 with high concurrency, range-read
     each requesting patch from the warm URL, write ``*_udm2.tif`` next
     to the SR.
  4. Sleep ``--poll-seconds`` and repeat until ``--idle-checks`` passes
     find nothing to do.

Idempotent: skips any patch whose UDM2 already exists.

Example:
    uv run scripts/udm2_fill.py --concurrency 64 --poll-seconds 120 --idle-checks 3
"""

import argparse
import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import rasterio
import rasterio.windows
from dotenv import load_dotenv
from planet import Session
from rich.logging import RichHandler

from ftw_planet.planet import (
    ASSET_UDM2,
    activate_asset_url,
    aoi_bounds_in_target,
    cog_env,
    require_api_key,
)

log = logging.getLogger("ftw_planet.udm2_fill")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument(
        "--idle-checks",
        type=int,
        default=3,
        help="Stop after this many consecutive passes with no SR-only patches found.",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=0,
        help="Hard cap on pass count (0 = unlimited).",
    )
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
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


def _index_item_ids(planet_root: Path) -> dict[tuple[str, str, str], str]:
    """Map (country, id, window) -> latest item_id from resample_log + extract shards."""
    out: dict[tuple[str, str, str], str] = {}
    for source in sorted((planet_root / "_global" / "extract").glob("shard_*.jsonl")) + [
        planet_root / "_global" / "resample_log.jsonl"
    ]:
        for r in _read_jsonl(source):
            if r.get("status") not in ("matched", "skipped_existing"):
                continue
            iid = r.get("item_id")
            if not iid:
                continue
            k = (str(r.get("country", "")), str(r.get("id", "")), str(r.get("window", "")))
            out[k] = str(iid)
    return out


def _index_geometries(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for r in _read_jsonl(planet_root / "_global" / "manifest.jsonl"):
        out[(r["country"], r["id"], r["window"])] = r["geometry_4326"]
    return out


def _find_sr_without_udm2(planet_root: Path) -> list[tuple[str, str, str, Path]]:
    """Walk per-country dirs; return SR tifs missing their UDM2 partner."""
    todo: list[tuple[str, str, str, Path]] = []
    for d in sorted(planet_root.iterdir()):
        if not d.is_dir() or d.name == "_global":
            continue
        country = d.name
        for sr in d.glob("*_a.tif"):
            if "_udm2" in sr.name or "_label" in sr.name:
                continue
            stem = sr.stem  # e.g. 1592589_a
            udm2 = d / f"{stem}_udm2.tif"
            if udm2.exists():
                continue
            pid, _, win = stem.rpartition("_")
            todo.append((country, pid, win, sr))
        for sr in d.glob("*_b.tif"):
            if "_udm2" in sr.name or "_label" in sr.name:
                continue
            stem = sr.stem
            udm2 = d / f"{stem}_udm2.tif"
            if udm2.exists():
                continue
            pid, _, win = stem.rpartition("_")
            todo.append((country, pid, win, sr))
    return todo


def _read_udm2_window_and_write(udm2_url: str, geom: dict, out_path: Path) -> None:
    """Range-read patch AOI from UDM2 COG, write local uint8 8-band tif."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with cog_env(), rasterio.open(udm2_url) as src:
        src_crs = src.crs.to_string()
        bounds = aoi_bounds_in_target(geom, src_crs)
        w = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        col = math.floor(w.col_off)
        row = math.floor(w.row_off)
        width = math.ceil(w.col_off + w.width) - col
        height = math.ceil(w.row_off + w.height) - row
        win = rasterio.windows.Window.from_slices(
            (row, row + height), (col, col + width), boundless=True
        )
        data = src.read(window=win)
        transform = src.window_transform(win)
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src_crs,
            "transform": transform,
            "compress": "ZSTD",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "predictor": 1,
            "nodata": src.nodata,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            if src.descriptions:
                dst.descriptions = src.descriptions


async def _process_pass(sess: Session, args: argparse.Namespace, log_path: Path) -> int:
    """Return number of UDM2 tifs successfully written in this pass."""
    item_ids = _index_item_ids(args.planet_root)
    geoms = _index_geometries(args.planet_root)
    todo = _find_sr_without_udm2(args.planet_root)
    if not todo:
        return 0

    # Build {item_id: [(country, pid, win, geom)]}
    by_scene: dict[str, list[tuple[str, str, str, dict]]] = {}
    skipped_missing = 0
    for country, pid, win, _sr in todo:
        key = (country, pid, win)
        iid = item_ids.get(key)
        geom = geoms.get(key)
        if not iid or not geom:
            skipped_missing += 1
            continue
        by_scene.setdefault(iid, []).append((country, pid, win, geom))
    log.info(
        "pass: %d SR-only patches; %d unique scenes; %d skipped (no item_id or geom)",
        len(todo),
        len(by_scene),
        skipped_missing,
    )

    if not by_scene:
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    written = 0

    async def _do_scene(iid: str, members: list[tuple[str, str, str, dict]]) -> int:
        async with sem:
            t = time.perf_counter()
            try:
                udm2_url = await activate_asset_url(sess, iid, ASSET_UDM2)
            except Exception as e:
                _append(
                    log_path, {"item_id": iid, "status": "udm2_activate_failed", "error": str(e)}
                )
                return 0
            n_ok = 0
            for country, pid, win, geom in members:
                out_path = args.planet_root / country / f"{pid}_{win}_udm2.tif"
                if out_path.exists():
                    continue
                try:
                    await asyncio.to_thread(_read_udm2_window_and_write, udm2_url, geom, out_path)
                    n_ok += 1
                    _append(
                        log_path,
                        {
                            "item_id": iid,
                            "country": country,
                            "id": pid,
                            "window": win,
                            "status": "filled",
                        },
                    )
                except Exception as e:
                    _append(
                        log_path,
                        {
                            "item_id": iid,
                            "country": country,
                            "id": pid,
                            "window": win,
                            "status": "fill_failed",
                            "error": str(e),
                        },
                    )
            log.info(
                "scene %s -> %d/%d UDM2 written (wall %.1fs)",
                iid,
                n_ok,
                len(members),
                time.perf_counter() - t,
            )
            return n_ok

    coros = [_do_scene(iid, members) for iid, members in by_scene.items()]
    for fut in asyncio.as_completed(coros):
        written += await fut
    return written


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    log_path = args.planet_root / "_global" / "udm2_fill_log.jsonl"
    idle = 0
    passes = 0
    async with Session() as sess:
        while True:
            passes += 1
            n_written = await _process_pass(sess, args, log_path)
            log.info("pass %d: wrote %d UDM2 tifs", passes, n_written)
            if n_written == 0:
                idle += 1
                if idle >= args.idle_checks:
                    log.info("idle for %d consecutive passes — stopping", idle)
                    break
            else:
                idle = 0
            if args.max_passes and passes >= args.max_passes:
                log.info("hit max-passes=%d — stopping", args.max_passes)
                break
            log.info("sleeping %ds before next pass", args.poll_seconds)
            await asyncio.sleep(args.poll_seconds)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_time=True)]
    )
    for noisy in ("planet", "httpx", "httpcore", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
