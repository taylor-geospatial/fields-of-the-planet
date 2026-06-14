"""Phase 1 — search PSScene candidates for one shard of the manifest.

Reads ``<out>/_global/manifest.jsonl``, processes rows where
``idx % num_shards == shard_id``, writes
``<out>/_global/search/shard_<shard_id>.jsonl``.

Idempotent: rows already in the shard's output are skipped on rerun.

Example:
    uv run scripts/search_shard.py --out data/planet \
        --shard-id 7 --num-shards 32 --concurrency 64 --max-cloud-cover 0.1
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from planet import Session
from rich.logging import RichHandler

from ftw_planet.planet import require_api_key, search_best_scene

log = logging.getLogger("ftw_planet.search_shard")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--search-days", type=int, default=0)
    p.add_argument("--max-cloud-cover", type=float, default=0.1)
    p.add_argument("--min-coverage", type=float, default=0.99)
    return p.parse_args()


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


async def _search_one(
    sess: Session,
    row: dict,
    *,
    user_search_days: int,
    max_cloud_cover: float,
    min_coverage: float,
) -> dict:
    target_date = datetime.fromisoformat(row["ftw_date"])
    rng = row["range"]
    auto_days = max(
        1,
        (datetime.fromisoformat(rng[1]) - datetime.fromisoformat(rng[0])).days // 2 + 1,
    )
    search_days = user_search_days if user_search_days > 0 else auto_days
    geom = row["geometry_4326"]

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
        "country": row["country"],
        "id": row["id"],
        "window": row["window"],
        "ftw_date": row["ftw_date"],
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


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    manifest_path: Path = args.out / "_global" / "manifest.jsonl"
    out_dir: Path = args.out / "_global" / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{args.shard_id:03d}.jsonl"

    # Read manifest, filter to this shard.
    rows: list[dict] = []
    with manifest_path.open() as f:
        for idx, line in enumerate(f):
            if idx % args.num_shards == args.shard_id:
                rows.append(json.loads(line))
    log.info("shard %d/%d: %d rows", args.shard_id, args.num_shards, len(rows))

    done = {(r["country"], r["id"], r["window"]) for r in _read_jsonl(shard_path)}
    todo = [r for r in rows if (r["country"], r["id"], r["window"]) not in done]
    log.info("  %d cached, %d to search", len(done), len(todo))

    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(r: dict) -> dict:
        async with sem:
            return await _search_one(
                sess,
                r,
                user_search_days=args.search_days,
                max_cloud_cover=args.max_cloud_cover,
                min_coverage=args.min_coverage,
            )

    async with Session() as sess:
        coros = [_wrapped(r) for r in todo]
        with shard_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                f.write(json.dumps(row) + "\n")
                f.flush()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    for noisy in ("planet", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
