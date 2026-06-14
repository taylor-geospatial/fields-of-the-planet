"""3-phase resample for bad PlanetScope patches — global batching beats per-patch.

The original ``resample_cloudy.py`` activated UDM2 per patch sequentially,
which forces a cold-storage thaw inside every per-patch lookup. This
version applies the main pipeline's pattern to alternative-scene
recovery:

  1. **Search** — for every bad (patch, window), find up to ``--max-candidates``
     candidate PSScene IDs within the season window, ranked by date and
     scene cloud cover. Output: ``_global/resample/search.jsonl``.

  2. **Probe** — collect unique candidate item_ids globally, activate UDM2
     for each *exactly once* with high concurrency, then for each
     candidate's UDM2 COG read the AOI of every requesting patch and
     compute per-band quality. Output:
     ``_global/resample/probes.jsonl`` (one row per (patch, candidate)).

  3. **Pick + Extract** — for each patch, choose the candidate with the
     best patch-level quality that passes thresholds; group winners by
     scene; activate SR for each unique winner once; range-read every
     winner's patches in scene-grouped mode and overwrite the local SR +
     UDM2. Output: ``_global/resample/picks.jsonl`` and
     ``_global/resample_log.jsonl``.

Each phase is resumable via its JSONL cache.

Example:
    uv run scripts/resample_v2.py --phase all \
        --max-candidates 5 --concurrency 64
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import rasterio
import rasterio.windows
from dotenv import load_dotenv
from planet import Session, data_filter
from rich.logging import RichHandler
from shapely.geometry import shape

from ftw_planet.pipeline import _extract_scene_group
from ftw_planet.planet import (
    ASSET_SR,
    ASSET_UDM2,
    ITEM_TYPE,
    activate_asset_url,
    aoi_bounds_in_target,
    cog_env,
    require_api_key,
)

log = logging.getLogger("ftw_planet.resample_v2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--phase", choices=("all", "search", "probe", "extract"), default="all")
    p.add_argument(
        "--probe-udm2",
        action="store_true",
        help="Activate UDM2 per candidate to verify patch-level quality before extract. "
        "Default off — trust scene-level cloud_cover and pick candidate #1; UDM2 still "
        "activated for the WINNER and saved alongside SR. Subsequent udm2_quality + "
        "another resample pass can catch any patches that slip through.",
    )
    # Quality thresholds
    p.add_argument("--min-clear", type=float, default=0.95)
    p.add_argument("--max-cloud", type=float, default=0.05)
    p.add_argument("--max-shadow", type=float, default=0.05)
    p.add_argument("--max-light-haze", type=float, default=0.10)
    p.add_argument("--max-heavy-haze", type=float, default=0.05)
    p.add_argument("--max-snow", type=float, default=0.05)
    p.add_argument("--max-unusable", type=float, default=0.05)
    # Search params
    p.add_argument("--scene-max-cloud", type=float, default=0.40)
    p.add_argument("--search-days", type=int, default=60)
    p.add_argument("--max-candidates", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=64)
    return p.parse_args()


QUALITY_BANDS = (
    ("cloud", "max_cloud"),
    ("shadow", "max_shadow"),
    ("light_haze", "max_light_haze"),
    ("heavy_haze", "max_heavy_haze"),
    ("snow", "max_snow"),
    ("unusable", "max_unusable"),
)


def _is_bad(q: dict, args: argparse.Namespace) -> bool:
    if q.get("clear", 1.0) < args.min_clear:
        return True
    return any(q.get(b, 0.0) > getattr(args, attr) for b, attr in QUALITY_BANDS)


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


def _build_bad_list(args: argparse.Namespace) -> list[dict]:
    """Same logic as resample_cloudy._load_bad_patches: union of UDM2-bad + extract-bad."""
    g = args.planet_root / "_global"
    seen: set[tuple[str, str, str]] = set()
    bad: list[dict] = []
    # UDM2-bad
    for r in _read_jsonl(g / "udm2_quality.jsonl"):
        if "clear" not in r:
            continue
        if _is_bad(r, args):
            key = (r["country"], r["id"], r["window"])
            seen.add(key)
            bad.append({**r, "_source": "udm2"})
    # Extract-bad (no_url / open_failed / extract_failed / scene_failed)
    fail = {"no_url", "open_failed", "extract_failed", "scene_failed"}
    latest: dict[tuple[str, str, str], dict] = {}
    for shard in sorted((g / "extract").glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            k = (str(r.get("country", "")), str(r.get("id", "")), str(r.get("window", "")))
            latest[k] = r
    for key, r in latest.items():
        if key in seen:
            continue
        if r.get("status") not in fail:
            continue
        bad.append(
            {
                "country": key[0],
                "id": key[1],
                "window": key[2],
                "_source": "extract",
            }
        )
    return bad


def _index_search(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for shard in sorted((planet_root / "_global" / "search").glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") != "found":
                continue
            out[(r["country"], r["id"], r["window"])] = r
    return out


def _index_manifest(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for r in _read_jsonl(planet_root / "_global" / "manifest.jsonl"):
        out[(r["country"], r["id"], r["window"])] = r
    return out


# ---------------------------------------------------------------------------
# Phase 1 — search
# ---------------------------------------------------------------------------


async def _search_one(
    sess: Session,
    item_to_exclude: str,
    geom: dict,
    target_date: datetime,
    args: argparse.Namespace,
) -> list[dict]:
    """Return up to ``max_candidates`` candidate scenes."""
    cl: Any = sess.client("data")
    gte = (target_date - timedelta(days=args.search_days)).replace(tzinfo=UTC)
    lte = (target_date + timedelta(days=args.search_days)).replace(tzinfo=UTC)
    sf = data_filter.and_filter(
        [
            data_filter.geometry_filter(geom),
            data_filter.date_range_filter("acquired", gte=gte, lte=lte),
            data_filter.range_filter("cloud_cover", lte=args.scene_max_cloud),
            data_filter.permission_filter(),
            data_filter.std_quality_filter(),
        ]
    )
    pgeom = shape(geom)
    parea = max(pgeom.area, 1e-12)
    cands: list[dict] = []
    async for it in cl.search([ITEM_TYPE], search_filter=sf, limit=200):
        if it["id"] == item_to_exclude:
            continue
        try:
            sg = shape(it["geometry"])
        except (KeyError, ValueError):
            continue
        cov = pgeom.intersection(sg).area / parea
        if cov < 0.99:
            continue
        props = it["properties"]
        cands.append(
            {
                "item_id": it["id"],
                "acquired": props["acquired"],
                "cloud_cover": float(props.get("cloud_cover", 1.0)),
                "coverage": cov,
            }
        )
    target_utc = target_date.replace(tzinfo=UTC) if target_date.tzinfo is None else target_date

    def _score(c: dict) -> tuple:
        a = datetime.fromisoformat(c["acquired"])
        return (abs((a - target_utc).total_seconds()), c["cloud_cover"])

    cands.sort(key=_score)
    return cands[: args.max_candidates]


async def phase_search(args: argparse.Namespace) -> None:
    g = args.planet_root / "_global" / "resample"
    g.mkdir(parents=True, exist_ok=True)
    out_path = g / "search.jsonl"
    done = {(r["country"], r["id"], r["window"]) for r in _read_jsonl(out_path)}

    bad = _build_bad_list(args)
    todo = [b for b in bad if (b["country"], b["id"], b["window"]) not in done]
    log.info("phase1 search: %d cached, %d to search", len(done), len(todo))
    if not todo:
        return

    search_idx = _index_search(args.planet_root)
    manifest = _index_manifest(args.planet_root)

    sem = asyncio.Semaphore(args.concurrency)

    async def _wrapped(b: dict) -> dict:
        async with sem:
            key = (b["country"], b["id"], b["window"])
            sr = search_idx.get(key)
            mr = manifest.get(key)
            if not sr or not mr:
                return {
                    **{k: b[k] for k in ("country", "id", "window")},
                    "candidates": [],
                    "status": "missing_indices",
                }
            geom = mr["geometry_4326"]
            target_date = datetime.fromisoformat(sr["scene_date"])
            cands = await _search_one(sess, sr["item_id"], geom, target_date, args)
            return {
                "country": b["country"],
                "id": b["id"],
                "window": b["window"],
                "previous_item_id": sr["item_id"],
                "candidates": cands,
            }

    async with Session() as sess:
        coros = [_wrapped(b) for b in todo]
        with out_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                f.write(json.dumps(row) + "\n")
                f.flush()


# ---------------------------------------------------------------------------
# Phase 2 — probe (bulk activate UDM2 + range-read patch quality)
# ---------------------------------------------------------------------------


def _probe_quality(udm2_url: str, geom: dict) -> dict:
    """Range-read UDM2 over patch AOI, compute per-band fractions."""
    import math

    with cog_env(), rasterio.open(udm2_url) as src:
        bounds = aoi_bounds_in_target(geom, src.crs.to_string())
        w = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        col = math.floor(w.col_off)
        row = math.floor(w.row_off)
        width = math.ceil(w.col_off + w.width) - col
        height = math.ceil(w.row_off + w.height) - row
        win = rasterio.windows.Window.from_slices(
            (row, row + height), (col, col + width), boundless=True
        )
        d = src.read(window=win)
    n = d.shape[1] * d.shape[2]
    return {
        "clear": float((d[0] > 0).sum()) / n,
        "snow": float((d[1] > 0).sum()) / n,
        "shadow": float((d[2] > 0).sum()) / n,
        "light_haze": float((d[3] > 0).sum()) / n,
        "heavy_haze": float((d[4] > 0).sum()) / n,
        "cloud": float((d[5] > 0).sum()) / n,
        "unusable": float((d[7] > 0).sum()) / n,
    }


async def phase_probe(args: argparse.Namespace) -> None:
    g = args.planet_root / "_global" / "resample"
    search_path = g / "search.jsonl"
    probes_path = g / "probes.jsonl"
    udm2_urls_path = g / "udm2_activations.jsonl"

    search_rows = _read_jsonl(search_path)
    log.info("phase2 probe: %d search rows", len(search_rows))

    # Build {item_id: list of (patch, geom)}
    manifest = _index_manifest(args.planet_root)
    requests: dict[str, list[dict]] = {}
    for r in search_rows:
        for c in r.get("candidates", []):
            iid = c["item_id"]
            key = (r["country"], r["id"], r["window"])
            geom = manifest.get(key, {}).get("geometry_4326")
            if geom is None:
                continue
            requests.setdefault(iid, []).append(
                {
                    "country": r["country"],
                    "id": r["id"],
                    "window": r["window"],
                    "geometry_4326": geom,
                }
            )
    log.info(
        "  unique candidate scenes: %d (across %d (patch, candidate) pairs)",
        len(requests),
        sum(len(v) for v in requests.values()),
    )

    # Resume from cache
    udm2_done: dict[str, str | None] = {}
    for r in _read_jsonl(udm2_urls_path):
        udm2_done[r["item_id"]] = r.get("udm2_url")
    todo_ids = sorted(set(requests) - set(udm2_done))
    log.info("  %d UDM2 URLs cached, %d to activate", len(udm2_done), len(todo_ids))

    sem_act = asyncio.Semaphore(args.concurrency)

    async def _activate(iid: str) -> dict:
        async with sem_act:
            t = time.perf_counter()
            try:
                url = await activate_asset_url(sess, iid, ASSET_UDM2)
                err = None
            except Exception as e:
                url, err = None, str(e)
            return {
                "item_id": iid,
                "udm2_url": url,
                "udm2_error": err,
                "activate_s": round(time.perf_counter() - t, 3),
            }

    async with Session() as sess:
        if todo_ids:
            coros = [_activate(i) for i in todo_ids]
            with udm2_urls_path.open("a") as f:
                for fut in asyncio.as_completed(coros):
                    row = await fut
                    udm2_done[row["item_id"]] = row.get("udm2_url")
                    f.write(json.dumps(row) + "\n")
                    f.flush()

    # Probe quality for each (patch, candidate) where UDM2 URL is good.
    done_probes = {
        (r["country"], r["id"], r["window"], r["item_id"]) for r in _read_jsonl(probes_path)
    }

    async def _probe_pair(iid: str, member: dict) -> dict | None:
        key = (member["country"], member["id"], member["window"], iid)
        if key in done_probes:
            return None
        url = udm2_done.get(iid)
        if not url:
            return {
                "country": member["country"],
                "id": member["id"],
                "window": member["window"],
                "item_id": iid,
                "status": "no_udm2_url",
            }
        try:
            q = await asyncio.to_thread(_probe_quality, url, member["geometry_4326"])
            return {
                "country": member["country"],
                "id": member["id"],
                "window": member["window"],
                "item_id": iid,
                "status": "ok",
                **{k: round(v, 4) for k, v in q.items()},
            }
        except Exception as e:
            return {
                "country": member["country"],
                "id": member["id"],
                "window": member["window"],
                "item_id": iid,
                "status": "probe_failed",
                "error": str(e),
            }

    sem_probe = asyncio.Semaphore(args.concurrency)

    async def _wrapped_probe(iid: str, member: dict) -> dict | None:
        async with sem_probe:
            return await _probe_pair(iid, member)

    coros = [_wrapped_probe(iid, m) for iid, members in requests.items() for m in members]
    log.info("  probing %d (patch, candidate) pairs", len(coros))
    if coros:
        with probes_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                if row is None:
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()


# ---------------------------------------------------------------------------
# Phase 3 — pick winners + extract
# ---------------------------------------------------------------------------


async def phase_extract(args: argparse.Namespace) -> None:
    g = args.planet_root / "_global" / "resample"
    search_rows = _read_jsonl(g / "search.jsonl")
    manifest = _index_manifest(args.planet_root)

    picks_path = g / "picks.jsonl"
    log_path = args.planet_root / "_global" / "resample_log.jsonl"
    done_log = {(r["country"], r["id"], r["window"]) for r in _read_jsonl(log_path)}

    picks: list[dict] = []
    if args.probe_udm2:
        # Probe-driven pick: choose first candidate whose patch-level UDM2 passes.
        probes = _read_jsonl(g / "probes.jsonl")
        by_patch: dict[tuple[str, str, str], list[dict]] = {}
        for p in probes:
            if p.get("status") != "ok":
                continue
            by_patch.setdefault((p["country"], p["id"], p["window"]), []).append(p)
        for key, plist in by_patch.items():
            if key in done_log:
                continue
            passing = [p for p in plist if not _is_bad(p, args)]
            if passing:
                passing.sort(key=lambda x: x.get("clear", 0), reverse=True)
                picks.append({**passing[0], "_pick_reason": "passes_all"})
            else:
                best = max(plist, key=lambda x: x.get("clear", 0))
                if best.get("clear", 0) >= 0.50:
                    picks.append({**best, "_pick_reason": "best_partial"})
    else:
        # Naive pick: trust scene-level cloud_cover, take candidate #1 (already
        # sorted by Δdate, scene_cc during search). Skips per-candidate UDM2
        # activation entirely. Quality verified post-hoc by udm2_quality.py;
        # patches still failing thresholds become next-pass bad-patch candidates.
        for r in search_rows:
            key = (r["country"], r["id"], r["window"])
            if key in done_log:
                continue
            cands = r.get("candidates", [])
            if not cands:
                continue
            top = cands[0]
            picks.append(
                {
                    "country": r["country"],
                    "id": r["id"],
                    "window": r["window"],
                    "item_id": top["item_id"],
                    "_pick_reason": "scene_cc_top1",
                    "scene_cloud_cover": top["cloud_cover"],
                    "scene_acquired": top["acquired"],
                }
            )

    log.info("phase3 picks: %d patches with a viable winner", len(picks))

    if not picks_path.exists() or len(_read_jsonl(picks_path)) < len(picks):
        with picks_path.open("w") as f:
            for r in picks:
                f.write(json.dumps(r) + "\n")

    # Group picks by item_id, activate SR for each unique winner, scene-grouped extract.
    by_scene: dict[str, list[dict]] = {}
    for r in picks:
        if (r["country"], r["id"], r["window"]) in done_log:
            continue
        by_scene.setdefault(r["item_id"], []).append(r)
    log.info(
        "  %d unique winning scenes; %d patches to re-extract",
        len(by_scene),
        sum(len(v) for v in by_scene.values()),
    )

    if not by_scene:
        return

    # UDM2 URLs: from phase2 cache if it ran; otherwise activate fresh in parallel
    # with SR (for naive-pick path the probe phase was skipped, so no cache).
    udm2_urls = {r["item_id"]: r.get("udm2_url") for r in _read_jsonl(g / "udm2_activations.jsonl")}

    sr_urls_path = g / "sr_activations.jsonl"
    sr_done = {r["item_id"]: r.get("sr_url") for r in _read_jsonl(sr_urls_path)}

    sem = asyncio.Semaphore(args.concurrency)

    async def _activate_sr(iid: str) -> dict:
        async with sem:
            if iid in sr_done:
                return {"item_id": iid, "sr_url": sr_done[iid]}
            t = time.perf_counter()
            # SR only — UDM2 is filled in by scripts/udm2_fill.py afterward.
            # UDM2 activation is ~3x slower than SR, so skipping it here halves
            # the resample wall time.
            try:
                sr_url = await activate_asset_url(sess, iid, ASSET_SR)
                sr_err = None
            except Exception as e:
                sr_url, sr_err = None, str(e)
            return {
                "item_id": iid,
                "sr_url": sr_url,
                "sr_error": sr_err,
                "activate_s": round(time.perf_counter() - t, 3),
            }

    async with Session() as sess:
        coros = [_activate_sr(iid) for iid in by_scene if iid not in sr_done]
        with sr_urls_path.open("a") as f:
            for fut in asyncio.as_completed(coros):
                row = await fut
                sr_done[row["item_id"]] = row.get("sr_url")
                f.write(json.dumps(row) + "\n")
                f.flush()

        # Now scene-grouped extract.
        async def _do_scene(iid: str, members: list[dict]) -> list[dict]:
            sr_url = sr_done.get(iid)
            udm2_url = udm2_urls.get(iid)
            # Build per-country member lists with geometry from manifest.
            by_country: dict[str, list[dict]] = {}
            for m in members:
                key = (m["country"], m["id"], m["window"])
                geom = manifest.get(key, {}).get("geometry_4326")
                if geom is None:
                    continue
                by_country.setdefault(m["country"], []).append(
                    {"id": m["id"], "window": m["window"], "geometry_4326": geom}
                )
            results: list[dict] = []
            for country, ms in by_country.items():
                # Wipe existing files so _extract_scene_group re-extracts.
                out_dir = args.planet_root / country
                for m in ms:
                    (out_dir / f"{m['id']}_{m['window']}.tif").unlink(missing_ok=True)
                    (out_dir / f"{m['id']}_{m['window']}_udm2.tif").unlink(missing_ok=True)
                try:
                    rows = await asyncio.to_thread(
                        _extract_scene_group,
                        iid,
                        sr_url,
                        udm2_url,
                        ms,
                        out_dir,
                    )
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
                    results.append(row)
            return results

        coros2 = [_do_scene(iid, members) for iid, members in by_scene.items()]
        with log_path.open("a") as f:
            for fut in asyncio.as_completed(coros2):
                rows = await fut
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.flush()


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    if args.phase in ("all", "search"):
        await phase_search(args)
    if args.phase in ("all", "probe") and args.probe_udm2:
        await phase_probe(args)
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
