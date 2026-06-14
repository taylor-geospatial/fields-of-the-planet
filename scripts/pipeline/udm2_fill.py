"""3-phase UDM2 fill — search/activate/extract pattern from the main pipeline.

For every SR tif on disk lacking its ``_udm2.tif`` partner, run three
resumable phases:

  1. **Plan** — index (country, id, window) -> item_id from extract +
     resample logs; group missing patches by item_id; write
     ``_global/udm2_fill/plan.jsonl`` (one row per unique item_id with
     member patch list).

  2. **Activate** — for each item_id without a cached URL, call Planet
     ``:activate`` for UDM2 with high concurrency; write
     ``_global/udm2_fill/activations.jsonl``.

  3. **Extract** — for each scene group, range-read every requesting
     patch's window from the warm UDM2 URL; write
     ``<country>/<id>_<window>_udm2.tif`` and append per-patch result to
     ``_global/udm2_fill/extracts.jsonl``.

Each phase is idempotent and resumable from its JSONL cache.

Example:
    uv run scripts/udm2_fill_v2.py --phase all --concurrency 64
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

log = logging.getLogger("ftw_planet.udm2_fill_v2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--phase", choices=("all", "plan", "activate", "extract"), default="all")
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


def _index_item_ids(planet_root: Path) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    sources = sorted((planet_root / "_global" / "extract").glob("shard_*.jsonl"))
    sources += [planet_root / "_global" / "resample_log.jsonl"]
    sources += [planet_root / "_global" / "restore_log.jsonl"]
    for src in sources:
        for r in _read_jsonl(src):
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


def _find_sr_without_udm2(planet_root: Path) -> list[tuple[str, str, str]]:
    todo: list[tuple[str, str, str]] = []
    for d in sorted(planet_root.iterdir()):
        if not d.is_dir() or d.name == "_global":
            continue
        country = d.name
        for tif in d.iterdir():
            if "_udm2" in tif.name or "_label" in tif.name:
                continue
            if not tif.name.endswith(".tif"):
                continue
            stem = tif.stem
            pid, _, win = stem.rpartition("_")
            if win not in ("a", "b"):
                continue
            udm2 = d / f"{stem}_udm2.tif"
            if udm2.exists():
                continue
            todo.append((country, pid, win))
    return todo


# ---------------------------------------------------------------------------
# Phase 1 — plan
# ---------------------------------------------------------------------------


def phase_plan(args: argparse.Namespace) -> None:
    g = args.planet_root / "_global" / "udm2_fill"
    g.mkdir(parents=True, exist_ok=True)
    out_path = g / "plan.jsonl"

    item_ids = _index_item_ids(args.planet_root)
    geoms = _index_geometries(args.planet_root)
    todo = _find_sr_without_udm2(args.planet_root)
    log.info("found %d SR tifs lacking UDM2", len(todo))

    by_scene: dict[str, list[dict]] = {}
    skipped = 0
    for country, pid, win in todo:
        key = (country, pid, win)
        iid = item_ids.get(key)
        geom = geoms.get(key)
        if not iid or not geom:
            skipped += 1
            continue
        by_scene.setdefault(iid, []).append(
            {"country": country, "id": pid, "window": win, "geometry_4326": geom}
        )
    log.info(
        "plan: %d unique scenes; %d patches; %d skipped (no item_id/geom)",
        len(by_scene),
        sum(len(v) for v in by_scene.values()),
        skipped,
    )
    with out_path.open("w") as f:
        for iid, members in sorted(by_scene.items()):
            f.write(json.dumps({"item_id": iid, "members": members}) + "\n")
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Phase 2 — activate
# ---------------------------------------------------------------------------


async def _activate_one(sess: Session, iid: str) -> dict:
    row: dict[str, Any] = {"item_id": iid}
    t = time.perf_counter()
    try:
        row["udm2_url"] = await activate_asset_url(sess, iid, ASSET_UDM2)
    except Exception as e:
        row["udm2_url"] = None
        row["udm2_error"] = str(e)
    row["activate_s"] = round(time.perf_counter() - t, 3)
    return row


async def phase_activate(args: argparse.Namespace) -> None:
    require_api_key()
    g = args.planet_root / "_global" / "udm2_fill"
    plan = _read_jsonl(g / "plan.jsonl")
    acts_path = g / "activations.jsonl"
    done = {r["item_id"]: r for r in _read_jsonl(acts_path)}
    todo = [r["item_id"] for r in plan if r["item_id"] not in done]
    log.info("activate: %d cached, %d to activate", len(done), len(todo))
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str) -> dict:
        async with sem:
            return await _activate_one(sess, iid)

    async with Session() as sess:
        coros = [_wrapped(iid) for iid in todo]
        with acts_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                f.write(json.dumps(row) + "\n")
                f.flush()


# ---------------------------------------------------------------------------
# Phase 3 — extract
# ---------------------------------------------------------------------------


def _read_and_write_udm2(udm2_url: str, geom: dict, out_path: Path) -> None:
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


async def phase_extract(args: argparse.Namespace) -> None:
    g = args.planet_root / "_global" / "udm2_fill"
    plan = _read_jsonl(g / "plan.jsonl")
    urls = {r["item_id"]: r.get("udm2_url") for r in _read_jsonl(g / "activations.jsonl")}
    log_path = g / "extracts.jsonl"
    done = {
        (r["country"], r["id"], r["window"])
        for r in _read_jsonl(log_path)
        if r.get("status") == "filled"
    }

    sem = asyncio.Semaphore(args.concurrency)

    async def _do_scene(iid: str, members: list[dict]) -> int:
        udm2_url = urls.get(iid)
        if not udm2_url:
            return 0
        async with sem:
            n = 0
            for m in members:
                key = (m["country"], m["id"], m["window"])
                if key in done:
                    continue
                out_path = args.planet_root / m["country"] / f"{m['id']}_{m['window']}_udm2.tif"
                if out_path.exists():
                    n += 1
                    with log_path.open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "country": m["country"],
                                    "id": m["id"],
                                    "window": m["window"],
                                    "item_id": iid,
                                    "status": "skipped_existing",
                                }
                            )
                            + "\n"
                        )
                    continue
                try:
                    await asyncio.to_thread(
                        _read_and_write_udm2, udm2_url, m["geometry_4326"], out_path
                    )
                    n += 1
                    status = "filled"
                    err = None
                except Exception as e:
                    status = "fill_failed"
                    err = str(e)
                row: dict[str, Any] = {
                    "country": m["country"],
                    "id": m["id"],
                    "window": m["window"],
                    "item_id": iid,
                    "status": status,
                }
                if err:
                    row["error"] = err
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            return n

    coros = [_do_scene(r["item_id"], r["members"]) for r in plan]
    total = 0
    for fut in asyncio.as_completed(coros):
        total += await fut
    log.info("extract: %d UDM2 written", total)


async def _run(args: argparse.Namespace) -> None:
    if args.phase in ("all", "plan"):
        phase_plan(args)
    if args.phase in ("all", "activate"):
        await phase_activate(args)
    if args.phase in ("all", "extract"):
        await phase_extract(args)


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
