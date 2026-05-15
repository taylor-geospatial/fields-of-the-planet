"""Push remaining missing patches toward 100% coverage.

For every (country, id, window) where the local SR tif is missing, walk a
prioritised list of recovery strategies until one succeeds or we exhaust
options:

  1. **Original scene** — re-activate via the original item_id from
     ``_global/extract/shard_*.jsonl``. Often the original asset has
     recovered since the last attempt.
  2. **Resample candidate #1** — from
     ``_global/resample/search.jsonl`` (was picked the first time but
     produced a dead URL; retry in case Planet's queue has cleared).
  3. **Resample candidate #2..N** — remaining candidates from the
     resample search, not yet tried.
  4. **Fresh wider search** — call Data API again with a wider date
     window (defaults to season midpoint +/- 120 d) and looser scene
     cloud cap (default 0.70), grab up to ``--max-fresh-candidates``
     new options.

For each candidate the script activates SR (with retry-on-null), and on
success runs a single-patch scene-grouped extract. Result is logged to
``_global/completion_log.jsonl`` with status + which strategy worked.

Idempotent: skips patches whose SR already exists on disk.

Example:
    uv run scripts/patch_completion.py --concurrency 32 --fresh-search
"""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from planet import Session, data_filter
from rich.logging import RichHandler
from shapely.geometry import shape

from ftw_planet.pipeline import _extract_scene_group
from ftw_planet.planet import (
    ASSET_SR,
    ITEM_TYPE,
    activate_asset_url,
    require_api_key,
)

log = logging.getLogger("ftw_planet.completion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--fresh-search",
        action="store_true",
        help="Enable strategy 4: fresh wider Data API search.",
    )
    p.add_argument("--fresh-search-days", type=int, default=120)
    p.add_argument("--fresh-scene-max-cloud", type=float, default=0.70)
    p.add_argument("--max-fresh-candidates", type=int, default=10)
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


def _index_original_items(planet_root: Path) -> dict[tuple[str, str, str], str]:
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


def _index_resample_candidates(planet_root: Path) -> dict[tuple[str, str, str], list[str]]:
    out: dict[tuple[str, str, str], list[str]] = {}
    for r in _read_jsonl(planet_root / "_global" / "resample" / "search.jsonl"):
        key = (r["country"], r["id"], r["window"])
        cands = [c["item_id"] for c in r.get("candidates", [])]
        out[key] = cands
    return out


def _index_manifest(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for r in _read_jsonl(planet_root / "_global" / "manifest.jsonl"):
        out[(r["country"], r["id"], r["window"])] = r
    return out


def _find_missing(planet_root: Path, manifest: dict) -> list[dict]:
    missing: list[dict] = []
    for (country, pid, window), m in manifest.items():
        sr = planet_root / country / f"{pid}_{window}.tif"
        if sr.exists():
            continue
        missing.append(
            {
                "country": country,
                "id": pid,
                "window": window,
                "geometry_4326": m["geometry_4326"],
                "range": m["range"],
            }
        )
    return missing


async def _try_extract(
    sess: Session,
    iid: str,
    member: dict,
    planet_root: Path,
) -> tuple[bool, str | None]:
    """Activate SR for iid, range-read member's window. Returns (success, error_msg)."""
    try:
        sr_url = await activate_asset_url(sess, iid, ASSET_SR)
    except Exception as e:
        return False, f"activate: {e}"
    if not sr_url:
        return False, "null_url"
    out_dir = planet_root / member["country"]
    ms = [
        {"id": member["id"], "window": member["window"], "geometry_4326": member["geometry_4326"]}
    ]
    try:
        rows = await asyncio.to_thread(_extract_scene_group, iid, sr_url, None, ms, out_dir)
    except Exception as e:
        return False, f"extract: {e}"
    if not rows:
        return False, "empty"
    r = rows[0]
    return r.get("status") == "matched", r.get("error")


async def _fresh_search(
    sess: Session,
    geom: dict,
    season_range: tuple[str, str],
    args: argparse.Namespace,
) -> list[str]:
    cl: Any = sess.client("data")
    start = datetime.fromisoformat(season_range[0]).replace(tzinfo=UTC)
    end = datetime.fromisoformat(season_range[1]).replace(tzinfo=UTC)
    mid = start + (end - start) / 2
    gte = (mid - timedelta(days=args.fresh_search_days)).replace(tzinfo=UTC)
    lte = (mid + timedelta(days=args.fresh_search_days)).replace(tzinfo=UTC)
    sf = data_filter.and_filter(
        [
            data_filter.geometry_filter(geom),
            data_filter.date_range_filter("acquired", gte=gte, lte=lte),
            data_filter.range_filter("cloud_cover", lte=args.fresh_scene_max_cloud),
            data_filter.permission_filter(),
            data_filter.std_quality_filter(),
        ]
    )
    pgeom = shape(geom)
    parea = max(pgeom.area, 1e-12)
    cands: list[dict] = []
    async for it in cl.search([ITEM_TYPE], search_filter=sf, limit=200):
        try:
            sg = shape(it["geometry"])
        except (KeyError, ValueError):
            continue
        if pgeom.intersection(sg).area / parea < 0.99:
            continue
        props = it["properties"]
        cands.append(
            {
                "item_id": it["id"],
                "acquired": props["acquired"],
                "cloud_cover": float(props.get("cloud_cover", 1.0)),
            }
        )

    def _score(c: dict) -> tuple:
        a = datetime.fromisoformat(c["acquired"])
        return (abs((a - mid).total_seconds()), c["cloud_cover"])

    cands.sort(key=_score)
    return [c["item_id"] for c in cands[: args.max_fresh_candidates]]


async def _complete_one(
    sess: Session,
    member: dict,
    orig_iid: str | None,
    resample_cands: list[str],
    args: argparse.Namespace,
    log_path: Path,
) -> dict:
    tried: list[str] = []

    async def _attempt(iid: str, strategy: str) -> dict | None:
        ok, _err = await _try_extract(sess, iid, member, args.planet_root)
        tried.append(iid)
        if ok:
            return {
                **{k: member[k] for k in ("country", "id", "window")},
                "status": "completed",
                "item_id": iid,
                "strategy": strategy,
            }
        return None

    # Strategy 1: original
    if orig_iid and orig_iid not in tried:
        result = await _attempt(orig_iid, "original")
        if result:
            return result

    # Strategy 2-3: resample candidates
    for i, iid in enumerate(resample_cands):
        if iid in tried:
            continue
        result = await _attempt(iid, f"resample_cand_{i}")
        if result:
            return result

    # Strategy 4: fresh wider search
    if args.fresh_search:
        fresh = await _fresh_search(sess, member["geometry_4326"], tuple(member["range"]), args)
        for i, iid in enumerate(fresh):
            if iid in tried:
                continue
            result = await _attempt(iid, f"fresh_cand_{i}")
            if result:
                return result

    return {
        **{k: member[k] for k in ("country", "id", "window")},
        "status": "exhausted",
        "tried": tried,
    }


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    manifest = _index_manifest(args.planet_root)
    missing = _find_missing(args.planet_root, manifest)
    log.info("missing SR patches: %d", len(missing))
    if not missing:
        return

    orig_idx = _index_original_items(args.planet_root)
    resa_idx = _index_resample_candidates(args.planet_root)
    log_path = args.planet_root / "_global" / "completion_log.jsonl"

    # Resume: skip patches already in completion log as 'completed' or 'exhausted'
    done = {
        (r["country"], r["id"], r["window"])
        for r in _read_jsonl(log_path)
        if r.get("status") in ("completed", "exhausted")
    }
    todo = [m for m in missing if (m["country"], m["id"], m["window"]) not in done]
    log.info("  %d already in completion_log, %d to try", len(done), len(todo))
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(m: dict) -> dict:
        async with sem:
            key = (m["country"], m["id"], m["window"])
            return await _complete_one(
                sess, m, orig_idx.get(key), resa_idx.get(key, []), args, log_path
            )

    async with Session() as sess:
        coros = [_wrapped(m) for m in todo]
        with log_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                f.write(json.dumps(row) + "\n")
                f.flush()


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
