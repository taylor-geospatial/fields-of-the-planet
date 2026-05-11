"""Re-activate scenes whose activations.jsonl row has ``sr_url: null``.

These scenes returned no usable location URL on first activation (Planet
hiccup or activation race). The script:
  1. Walks ``activations.jsonl``, builds a per-item_id last-write-wins
     dict, and finds those still missing an ``sr_url`` (and/or
     ``udm2_url``) at the latest known state.
  2. Calls ``:activate`` for each missing asset.
  3. Appends fresh rows to ``activations.jsonl``. Extract loads the file
     as last-write-wins so the new URL replaces the null entry.

Idempotent: skips item_ids that have non-null URLs in their latest row.

Example:
    uv run scripts/reactivate_null_url.py --concurrency 32
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

log = logging.getLogger("ftw_planet.reactivate_null")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    p.add_argument("--concurrency", type=int, default=32)
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


async def _activate_one(sess: Session, item_id: str, need_sr: bool, need_udm2: bool) -> dict:
    row: dict[str, Any] = {"item_id": item_id, "_reactivated_null": True}

    async def _sr() -> tuple[str | None, str | None, float]:
        if not need_sr:
            return None, None, 0.0
        t = time.perf_counter()
        try:
            return await activate_asset_url(sess, item_id, ASSET_SR), None, time.perf_counter() - t
        except Exception as e:
            return None, str(e), time.perf_counter() - t

    async def _udm2() -> tuple[str | None, str | None, float]:
        if not need_udm2:
            return None, None, 0.0
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
    if need_sr:
        row["sr_url"] = sr_url
        if sr_err:
            row["sr_error"] = sr_err
        row["activate_sr_s"] = round(sr_s, 3)
    if need_udm2:
        row["udm2_url"] = udm2_url
        if udm2_err:
            row["udm2_error"] = udm2_err
        row["activate_udm2_s"] = round(udm2_s, 3)
    return row


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    activations_path: Path = args.out / "_global" / "activations.jsonl"
    rows = _read_jsonl(activations_path)
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["item_id"]] = r

    todo: list[tuple[str, bool, bool]] = []
    for iid, r in latest.items():
        need_sr = not r.get("sr_url")
        need_udm2 = not r.get("udm2_url")
        if need_sr or need_udm2:
            todo.append((iid, need_sr, need_udm2))
    log.info("activations rows: %d unique scenes; missing URLs: %d", len(latest), len(todo))
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str, need_sr: bool, need_udm2: bool) -> dict:
        async with sem:
            return await _activate_one(sess, iid, need_sr, need_udm2)

    async with Session() as sess:
        coros = [_wrapped(iid, ns, nu) for iid, ns, nu in todo]
        with activations_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                f.write(json.dumps(row) + "\n")
                f.flush()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_time=True)]
    )
    for noisy in ("planet", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
