"""Restore SR for patches that lost their original tif due to a failed resample.

For every (country, id, window) where the SR tif is missing on disk:
  1. Look up the **original** item_id from ``_global/extract/shard_*.jsonl``
     (status=matched|skipped_existing).
  2. Activate that scene's SR (Planet will hand back the cached URL if still
     warm, otherwise re-thaw).
  3. Range-read the patch window via scene-grouped extract.

Only restores SR (UDM2 will be filled later by udm2_fill.py).

Idempotent: skips patches whose SR already exists.

Example:
    uv run scripts/restore_lost_sr.py --concurrency 64
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from planet import Session
from rich.logging import RichHandler

from ftw_planet.pipeline import _extract_scene_group
from ftw_planet.planet import ASSET_SR, activate_asset_url, require_api_key

log = logging.getLogger("ftw_planet.restore_lost_sr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--concurrency", type=int, default=64)
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


def _original_item_ids(planet_root: Path) -> dict[tuple[str, str, str], str]:
    """Walk extract shards; map (country, id, window) -> original item_id."""
    out: dict[tuple[str, str, str], str] = {}
    for shard in sorted((planet_root / "_global" / "extract").glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
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


def _missing_sr_patches(planet_root: Path, item_idx: dict) -> dict[str, list[dict]]:
    """For every (country, id, window) in item_idx, check if SR exists; if not, group by item_id."""
    by_scene: dict[str, list[dict]] = {}
    for (country, pid, window), iid in item_idx.items():
        sr = planet_root / country / f"{pid}_{window}.tif"
        if sr.exists():
            continue
        by_scene.setdefault(iid, []).append({"country": country, "id": pid, "window": window})
    return by_scene


async def _do_scene(
    sess: Session,
    iid: str,
    members: list[dict],
    geoms: dict,
    planet_root: Path,
    log_path: Path,
) -> int:
    t = time.perf_counter()
    try:
        sr_url = await activate_asset_url(sess, iid, ASSET_SR)
    except Exception as e:
        with log_path.open("a") as f:
            f.write(
                json.dumps({"item_id": iid, "status": "activate_failed", "error": str(e)}) + "\n"
            )
        return 0
    if not sr_url:
        with log_path.open("a") as f:
            f.write(json.dumps({"item_id": iid, "status": "no_sr_url"}) + "\n")
        return 0

    by_country: dict[str, list[dict]] = {}
    for m in members:
        key = (m["country"], m["id"], m["window"])
        geom = geoms.get(key)
        if geom is None:
            continue
        by_country.setdefault(m["country"], []).append(
            {"id": m["id"], "window": m["window"], "geometry_4326": geom}
        )
    n_ok = 0
    rows_all: list[dict[str, Any]] = []
    for country, ms in by_country.items():
        out_dir = planet_root / country
        try:
            rows = await asyncio.to_thread(_extract_scene_group, iid, sr_url, None, ms, out_dir)
        except Exception as e:
            rows = [
                {
                    "id": m["id"],
                    "window": m["window"],
                    "item_id": iid,
                    "status": "scene_failed",
                    "error": str(e),
                }
                for m in ms
            ]
        for row in rows:
            row["country"] = country
            row["status_restore"] = (
                "restored" if row.get("status") == "matched" else row.get("status")
            )
            rows_all.append(row)
            if row.get("status") == "matched":
                n_ok += 1
    with log_path.open("a") as f:
        for row in rows_all:
            f.write(json.dumps(row) + "\n")
    log.info("%s: %d/%d restored (wall %.1fs)", iid, n_ok, len(members), time.perf_counter() - t)
    return n_ok


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    items = _original_item_ids(args.planet_root)
    geoms = _index_geometries(args.planet_root)
    by_scene = _missing_sr_patches(args.planet_root, items)
    log.info(
        "scenes needing restore: %d  patches: %d",
        len(by_scene),
        sum(len(v) for v in by_scene.values()),
    )
    if not by_scene:
        return

    log_path = args.planet_root / "_global" / "restore_log.jsonl"
    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str, members: list[dict]) -> int:
        async with sem:
            return await _do_scene(sess, iid, members, geoms, args.planet_root, log_path)

    async with Session() as sess:
        coros = [_wrapped(iid, members) for iid, members in by_scene.items()]
        total = 0
        for fut in asyncio.as_completed(coros):
            total += await fut
        log.info("restored %d patches across %d scenes", total, len(by_scene))


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
