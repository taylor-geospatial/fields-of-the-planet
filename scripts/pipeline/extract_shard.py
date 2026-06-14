"""Phase 3 — extract one shard of the global scene set.

Groups search-result rows by ``item_id`` (each scene = one group), assigns
groups to shards via ``hash(item_id) % num_shards == shard_id``. For each
scene the shard owns:
  - open the COG once with rasterio
  - read every member patch's window, write GeoTIFF to
    ``<out>/<country>/<patch>_<window>.tif`` (+ UDM2)
  - record per-extract timings to ``<out>/_global/extract/shard_<i>.jsonl``

Idempotent: rows already in this shard's log are skipped.

Example:
    uv run scripts/extract_shard.py --out data/planet \
        --shard-id 17 --num-shards 64 --concurrency 16
"""

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from ftw_planet.pipeline import _extract_scene_group  # internal helper

log = logging.getLogger("ftw_planet.extract_shard")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=16)
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


def _shard_for(item_id: str, num_shards: int) -> int:
    h = hashlib.sha1(item_id.encode()).digest()
    return int.from_bytes(h[:4], "big") % num_shards


def _country_out_dir(out_root: Path, country: str) -> Path:
    return out_root / country


async def _run(args: argparse.Namespace) -> None:
    g_dir: Path = args.out / "_global"
    search_dir = g_dir / "search"
    extract_dir = g_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    shard_path = extract_dir / f"shard_{args.shard_id:03d}.jsonl"

    # Load manifest to map (country, id, window) -> geometry_4326. Search shards
    # don't carry geometry forward to keep them small; we join here.
    manifest_path = g_dir / "manifest.jsonl"
    geom_by_key: dict[tuple[str, str, str], dict] = {}
    for r in _read_jsonl(manifest_path):
        geom_by_key[(r["country"], r["id"], r["window"])] = r["geometry_4326"]

    # Build {item_id: [member_rows]} for entries that hash to our shard.
    groups: dict[str, list[dict]] = defaultdict(list)
    for shard in sorted(search_dir.glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") != "found":
                continue
            iid = r["item_id"]
            if _shard_for(iid, args.num_shards) != args.shard_id:
                continue
            key = (r["country"], r["id"], r["window"])
            geom = geom_by_key.get(key)
            if geom is None:
                continue  # manifest mismatch; shouldn't happen
            r = {**r, "geometry_4326": geom}
            groups[iid].append(r)
    log.info(
        "shard %d/%d: %d scenes, %d members",
        args.shard_id,
        args.num_shards,
        len(groups),
        sum(len(v) for v in groups.values()),
    )

    # Look up activations.
    activations = {r["item_id"]: r for r in _read_jsonl(g_dir / "activations.jsonl")}

    # Skip only rows that were genuinely successful previously. no_url /
    # open_failed / extract_failed should retry next time round (their scenes
    # may now be activated, or the failure was transient).
    done = {
        (r["country"], r["id"], r["window"])
        for r in _read_jsonl(shard_path)
        if r.get("status") in ("matched", "skipped_existing")
    }
    log.info("  %d rows already successfully extracted, processing remaining", len(done))

    sem = asyncio.Semaphore(args.concurrency)

    async def _process_scene(iid: str, members: list[dict]) -> list[dict]:
        async with sem:
            # Filter out members already done.
            todo = [m for m in members if (m["country"], m["id"], m["window"]) not in done]
            if not todo:
                return []
            act = activations.get(iid, {})
            # Each member needs its country, plus geometry already in row.
            members_for_extract = [
                {"id": m["id"], "window": m["window"], "geometry_4326": m["geometry_4326"]}
                for m in todo
            ]
            # Country is uniform per member but we need per-member out dir.
            results: list[dict] = []
            # Group members by country so we can route outputs cleanly.
            by_country: dict[str, list[dict]] = defaultdict(list)
            for orig, m in zip(todo, members_for_extract, strict=False):
                by_country[orig["country"]].append(m)

            for country, ms in by_country.items():
                t0 = time.perf_counter()
                try:
                    rows = await asyncio.to_thread(
                        _extract_scene_group,
                        iid,
                        act.get("sr_url"),
                        act.get("udm2_url"),
                        ms,
                        _country_out_dir(args.out, country),
                    )
                except Exception as e:
                    log.warning("scene group failed for %s/%s: %s", iid, country, e)
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
                wall = round(time.perf_counter() - t0, 3)
                for row in rows:
                    row["country"] = country
                    row["scene_wall_s"] = wall
                    results.append(row)
            return results

    coros = [_process_scene(iid, members) for iid, members in groups.items()]
    if not coros:
        return

    with shard_path.open("a") as f:
        for fut in asyncio.as_completed(coros):
            rows = await fut
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    for noisy in ("planet", "httpx", "httpcore", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
