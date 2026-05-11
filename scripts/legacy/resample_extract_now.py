"""Extract resample patches whose SR has been activated so far.

Standalone extractor that pairs with resample_v2's running phase 3: it
processes scenes that already have an ``sr_url`` in
``_global/resample/sr_activations.jsonl`` and a winner row in
``_global/resample/picks.jsonl``. Designed to be run alongside the main
resample job so SR extraction happens incrementally instead of waiting
for every activation to land.

Idempotent: skips any (country, id, window) already logged as matched
in ``_global/resample_log.jsonl``. UDM2 is omitted here — the partner
``udm2_fill.py`` worker writes UDM2 from a separate activation pass.

Example:
    uv run scripts/resample_extract_now.py --concurrency 32
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from ftw_planet.pipeline import _extract_scene_group

log = logging.getLogger("ftw_planet.resa_extract_now")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
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


def _index_manifest(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for r in _read_jsonl(planet_root / "_global" / "manifest.jsonl"):
        out[(r["country"], r["id"], r["window"])] = r["geometry_4326"]
    return out


async def _do_scene(
    iid: str,
    sr_url: str,
    members: list[dict],
    planet_root: Path,
    geom_idx: dict,
    log_path: Path,
) -> list[dict]:
    by_country: dict[str, list[dict]] = {}
    for m in members:
        key = (m["country"], m["id"], m["window"])
        geom = geom_idx.get(key)
        if geom is None:
            continue
        by_country.setdefault(m["country"], []).append(
            {"id": m["id"], "window": m["window"], "geometry_4326": geom}
        )
    out_rows: list[dict] = []
    for country, ms in by_country.items():
        out_dir = planet_root / country
        # Don't wipe before extract: _extract_scene_group has skipped_existing
        # logic, but if we wiped and the new extract fails, we'd lose the
        # original SR entirely. Instead, force overwrite by deleting just
        # before each scene-group call, only after we know the URL is good.
        try:
            rows = await asyncio.to_thread(_extract_scene_group, iid, sr_url, None, ms, out_dir)
        except Exception as e:
            log.warning("scene extract failed for %s/%s: %s", iid, country, e)
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
            row["status_resample"] = (
                "matched_new" if row.get("status") == "matched" else row.get("status")
            )
            out_rows.append(row)
    with log_path.open("a") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    return out_rows


async def _run(args: argparse.Namespace) -> None:
    g = args.planet_root / "_global" / "resample"
    picks = _read_jsonl(g / "picks.jsonl")
    sr_rows = _read_jsonl(g / "sr_activations.jsonl")
    geom_idx = _index_manifest(args.planet_root)
    log_path = args.planet_root / "_global" / "resample_log.jsonl"
    done = {
        (r["country"], r["id"], r["window"])
        for r in _read_jsonl(log_path)
        if r.get("status_resample") == "matched_new" or r.get("status") == "matched"
    }

    sr_urls: dict[str, str] = {}
    for r in sr_rows:
        url = r.get("sr_url")
        if url:
            sr_urls[r["item_id"]] = url

    # Group picks by item_id, filtering to scenes with sr_url + unfinished members.
    by_scene: dict[str, list[dict]] = {}
    for p in picks:
        iid = p["item_id"]
        if iid not in sr_urls:
            continue
        if (p["country"], p["id"], p["window"]) in done:
            continue
        by_scene.setdefault(iid, []).append(p)

    log.info(
        "ready to extract: %d scenes / %d patches (sr_urls available)",
        len(by_scene),
        sum(len(v) for v in by_scene.values()),
    )
    if not by_scene:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(iid: str, members: list[dict]) -> int:
        async with sem:
            t = time.perf_counter()
            rows = await _do_scene(iid, sr_urls[iid], members, args.planet_root, geom_idx, log_path)
            n_ok = sum(1 for r in rows if r.get("status") == "matched")
            log.info("%s: %d/%d ok (%.1fs)", iid, n_ok, len(rows), time.perf_counter() - t)
            return n_ok

    coros = [_wrapped(iid, members) for iid, members in by_scene.items()]
    total_ok = 0
    for fut in asyncio.as_completed(coros):
        total_ok += await fut
    log.info("done: %d patches extracted across %d scenes", total_ok, len(by_scene))


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
