"""Match FTW patches to PlanetScope imagery in three phases.

Reads ``data/ftw/<country>/index.jsonl`` and runs:
  1. search    -> data/planet/<country>/search.jsonl       (per patch+window)
  2. activate  -> data/planet/<country>/activations.jsonl  (per unique scene)
  3. extract   -> data/planet/<country>/extracts.jsonl     (per patch+window)
                  + data/planet/<country>/<id>_<w>{,_udm2}.tif

Each phase is resumable — re-running skips already-cached rows. Different
concurrency knobs per phase let you tune for the workload (search is cheap
metadata, activate is API-bound, extract is HTTP-range network heavy).

Example:
    uv run scripts/match_planet.py --country rwanda \
        --search-concurrency 32 --activate-concurrency 16 --extract-concurrency 32 \
        --max-cloud-cover 0.1
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from planet import Session
from rich.logging import RichHandler

from ftw_planet.ftw import read_index
from ftw_planet.pipeline import activate_all, extract_all, search_all
from ftw_planet.planet import require_api_key

log = logging.getLogger("ftw_planet.match")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--country", required=True)
    p.add_argument("--ftw-root", type=Path, default=Path("data/ftw"))
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    p.add_argument(
        "--search-days",
        type=int,
        default=0,
        help="Half-window (days) around the FTW season midpoint. 0 = auto from season range.",
    )
    p.add_argument("--max-cloud-cover", type=float, default=0.1)
    p.add_argument("--min-coverage", type=float, default=0.99)
    p.add_argument("--search-concurrency", type=int, default=32)
    p.add_argument("--activate-concurrency", type=int, default=16)
    p.add_argument("--extract-concurrency", type=int, default=32)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N patches (0 = all). Useful for smoke tests.",
    )
    p.add_argument(
        "--phase",
        choices=("all", "search", "activate", "extract"),
        default="all",
        help="Run only one phase. Useful for profiling or partial reruns.",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    index_path: Path = args.ftw_root / args.country / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found — run download_ftw.py first")

    patches = read_index(index_path)
    if args.limit > 0:
        patches = patches[: args.limit]
    patches_by_id = {p["id"]: p for p in patches}
    log.info("loaded %d patches from %s", len(patches), index_path)

    out_dir: Path = args.out / args.country
    out_dir.mkdir(parents=True, exist_ok=True)
    search_cache = out_dir / "search.jsonl"
    act_cache = out_dir / "activations.jsonl"
    ext_cache = out_dir / "extracts.jsonl"

    async with Session() as sess:
        # Phase 1
        if args.phase in ("all", "search"):
            t0 = time.perf_counter()
            search_rows = await search_all(
                sess,
                patches,
                cache_path=search_cache,
                concurrency=args.search_concurrency,
                max_cloud_cover=args.max_cloud_cover,
                min_coverage=args.min_coverage,
                search_days=args.search_days,
            )
            log.info("phase1 wall: %.1fs", time.perf_counter() - t0)
        else:
            from ftw_planet.pipeline import _read_jsonl

            search_rows = _read_jsonl(search_cache)

        if args.phase == "search":
            return

        item_ids = [r["item_id"] for r in search_rows if r.get("status") == "found"]

        # Phase 2
        if args.phase in ("all", "activate"):
            t0 = time.perf_counter()
            activations = await activate_all(
                sess,
                item_ids,
                cache_path=act_cache,
                concurrency=args.activate_concurrency,
            )
            log.info("phase2 wall: %.1fs", time.perf_counter() - t0)
        else:
            from ftw_planet.pipeline import _read_jsonl

            activations = {r["item_id"]: r for r in _read_jsonl(act_cache)}

        if args.phase == "activate":
            return

        # Phase 3
        t0 = time.perf_counter()
        extract_rows = await extract_all(
            search_rows,
            patches_by_id,
            activations,
            out_dir=out_dir,
            cache_path=ext_cache,
            concurrency=args.extract_concurrency,
        )
        log.info("phase3 wall: %.1fs", time.perf_counter() - t0)

        # Summary
        from collections import Counter

        statuses = Counter(r.get("status") for r in extract_rows)
        log.info("extract statuses: %s", dict(statuses))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True)],
    )
    # Quiet noisy planet/httpx INFO chatter; keep our pipeline logs.
    for noisy in ("planet", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_dotenv()
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
