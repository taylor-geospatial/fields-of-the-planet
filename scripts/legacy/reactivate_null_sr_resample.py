"""Re-activate SR for resample winners whose sr_url came back null.

Scans ``_global/resample/sr_activations.jsonl`` for rows where ``sr_url`` is
null/missing, re-calls Planet ``:activate`` for those item_ids, and appends
fresh rows. ``resample_extract_now.py`` then sees the latest-write-wins
sr_url and can extract those previously-failed scenes.

Idempotent: re-runs will retry remaining null-URL scenes.

Example:
    uv run scripts/reactivate_null_sr_resample.py --concurrency 64
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

from ftw_planet.planet import ASSET_SR, activate_asset_url, require_api_key

log = logging.getLogger("ftw_planet.reactivate_null_sr_resample")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/planet"))
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


async def _activate(sess: Session, iid: str) -> dict:
    row: dict[str, Any] = {"item_id": iid, "_reactivated_null_sr": True}
    t = time.perf_counter()
    try:
        row["sr_url"] = await activate_asset_url(sess, iid, ASSET_SR)
    except Exception as e:
        row["sr_url"] = None
        row["sr_error"] = str(e)
    row["activate_sr_s"] = round(time.perf_counter() - t, 3)
    return row


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    sr_path = args.out / "_global" / "resample" / "sr_activations.jsonl"
    rows = _read_jsonl(sr_path)
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["item_id"]] = r
    todo = [iid for iid, r in latest.items() if not r.get("sr_url")]
    log.info("total scenes: %d; missing sr_url: %d", len(latest), len(todo))
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str) -> dict:
        async with sem:
            return await _activate(sess, iid)

    async with Session() as sess:
        coros = [_wrapped(iid) for iid in todo]
        with sr_path.open("a") as f:
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
