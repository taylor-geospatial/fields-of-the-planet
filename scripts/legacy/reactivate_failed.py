"""Re-activate scenes that previously yielded broken (HTTP 400) URLs.

Reads extract shard logs, finds item_ids with any ``open_failed`` rows,
re-calls Planet ``:activate`` on each (gets a fresh signed URL), then
appends the new row to ``activations.jsonl``. Extract loads activations
as a dict keyed by item_id with last-write-wins, so the new URL replaces
the broken one on the next extract pass.

Example:
    uv run scripts/reactivate_failed.py --concurrency 32
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

log = logging.getLogger("ftw_planet.reactivate")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    p.add_argument("--concurrency", type=int, default=32)
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
    row: dict[str, Any] = {"item_id": item_id, "_reactivated": True}

    async def _sr() -> tuple[str | None, str | None, float]:
        t = time.perf_counter()
        try:
            return await activate_asset_url(sess, item_id, ASSET_SR), None, time.perf_counter() - t
        except Exception as e:
            return None, str(e), time.perf_counter() - t

    async def _udm2() -> tuple[str | None, str | None, float]:
        t = time.perf_counter()
        try:
            return (
                await activate_asset_url(sess, item_id, ASSET_UDM2),
                None,
                time.perf_counter() - t,
            )
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
    g_dir: Path = args.out / "_global"
    extract_dir = g_dir / "extract"
    activations_path = g_dir / "activations.jsonl"

    # Find item_ids with at least one open_failed extract row.
    bad: set[str] = set()
    for shard in sorted(extract_dir.glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") == "open_failed":
                bad.add(r["item_id"])
    log.info("found %d unique scenes with open_failed history", len(bad))

    # Don't bother reactivating ones that have a successful follow-up matched row.
    good_followup: set[str] = set()
    for shard in sorted(extract_dir.glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") in ("matched", "skipped_existing"):
                good_followup.add(r["item_id"])
    todo = sorted(bad - good_followup)
    log.info("after dropping scenes with later success: %d to reactivate", len(todo))

    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str) -> dict:
        async with sem:
            return await _activate_one(sess, iid)

    async with Session() as sess:
        coros = [_wrapped(iid) for iid in todo]
        with activations_path.open("a") as f:
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
