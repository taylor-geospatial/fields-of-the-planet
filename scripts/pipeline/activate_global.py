"""Phase 2 — globally dedup item_ids from all search shards and activate each.

Reads every ``<out>/_global/search/shard_*.jsonl``, collects unique item_ids,
activates SR + UDM2 per scene, writes ``<out>/_global/activations.jsonl``.

Idempotent: scenes already in activations.jsonl are skipped.

Example:
    uv run scripts/activate_global.py --out data/planet --concurrency 64
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

from ftw_planet.planet import (
    ASSET_SR,
    ASSET_UDM2,
    activate_asset_url,
    require_api_key,
)

log = logging.getLogger("ftw_planet.activate")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    p.add_argument("--concurrency", type=int, default=64)
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


async def _activate_one(sess: Session, item_id: str) -> dict:
    """Activate SR + UDM2 in parallel — they share no state on Planet's side."""
    row: dict[str, Any] = {"item_id": item_id}

    async def _sr() -> tuple[str | None, str | None, float]:
        t = time.perf_counter()
        try:
            url = await activate_asset_url(sess, item_id, ASSET_SR)
            return url, None, time.perf_counter() - t
        except Exception as e:
            return None, str(e), time.perf_counter() - t

    async def _udm2() -> tuple[str | None, str | None, float]:
        t = time.perf_counter()
        try:
            url = await activate_asset_url(sess, item_id, ASSET_UDM2)
            return url, None, time.perf_counter() - t
        except Exception as e:
            return None, str(e), time.perf_counter() - t

    (sr_url, sr_err, sr_s), (udm2_url, udm2_err, udm2_s) = await asyncio.gather(_sr(), _udm2())
    row["sr_url"] = sr_url
    if sr_err:
        row["sr_error"] = sr_err
    row["activate_sr_s"] = round(sr_s, 3)
    row["udm2_url"] = udm2_url
    if udm2_err:
        row["udm2_error"] = udm2_err
    row["activate_udm2_s"] = round(udm2_s, 3)
    return row


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    search_dir: Path = args.out / "_global" / "search"
    out_path: Path = args.out / "_global" / "activations.jsonl"

    # Collect unique item_ids across all search shards.
    item_ids: set[str] = set()
    for shard in sorted(search_dir.glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") == "found":
                item_ids.add(r["item_id"])
    log.info("collected %d unique scenes from %s", len(item_ids), search_dir)

    done = {r["item_id"] for r in _read_jsonl(out_path)}
    todo = sorted(item_ids - done)
    log.info("  %d cached, %d to activate", len(done), len(todo))

    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str) -> dict:
        async with sem:
            return await _activate_one(sess, iid)

    async with Session() as sess:
        coros = [_wrapped(iid) for iid in todo]
        with out_path.open("a") as f:
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
