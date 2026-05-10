"""Re-sample patches whose existing PlanetScope image is too cloudy/unusable.

Reads ``data/planet/_global/udm2_quality.jsonl`` (produced by
``udm2_quality.py``), filters patches with ``clear < min_clear`` or
``unusable > max_unusable``, then for each bad patch:

  1. Searches the Data API for **alternative** PSScene candidates within a
     wider date window and higher scene-cloud-cover ceiling. Excludes the
     scene already used (``current_item_id``).
  2. For each candidate ranked by (Δdate, scene cloud_cover), activates
     UDM2 only and range-reads the patch-area to compute *patch-level*
     cloud / clear / unusable. Stops on the first candidate that meets the
     quality threshold.
  3. Activates the SR asset for the winning scene, range-reads the patch,
     and overwrites the existing GeoTIFF + UDM2.
  4. Logs the swap to ``data/planet/_global/resample_log.jsonl``.

Strategy choices:
  * Scene-level cloud_cover up to ``--scene-max-cloud`` (default 0.40) at
    search time — we'll filter patch-level afterwards.
  * Up to ``--max-candidates`` (default 5) UDM2 probes per patch before
    giving up.
  * Per-scene UDM2 cache so many patches sharing one alternative scene
    only probe it once.

Idempotent: a patch already in ``resample_log.jsonl`` is skipped.

Example:
    uv run scripts/resample_cloudy.py --min-clear 0.95 --max-unusable 0.05 \
        --scene-max-cloud 0.4 --max-candidates 5 --concurrency 16
"""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

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

log = logging.getLogger("ftw_planet.resample")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    # Aggregate quality
    p.add_argument(
        "--min-clear", type=float, default=0.95, help="Min fraction of clear pixels required."
    )
    # Per-band ceilings (all UDM2 categories — anything above these triggers resample)
    p.add_argument("--max-cloud", type=float, default=0.05)
    p.add_argument("--max-shadow", type=float, default=0.05)
    p.add_argument("--max-light-haze", type=float, default=0.10)
    p.add_argument("--max-heavy-haze", type=float, default=0.05)
    p.add_argument("--max-snow", type=float, default=0.05)
    p.add_argument("--max-unusable", type=float, default=0.05)
    # Search params
    p.add_argument("--scene-max-cloud", type=float, default=0.40)
    p.add_argument(
        "--search-days",
        type=int,
        default=0,
        help="Half-window around season midpoint. 0=auto (full season).",
    )
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=16)
    return p.parse_args()


# Bands and their per-row JSONL keys, paired with the args attribute that
# defines the per-band ceiling. Used both to flag bad patches at filter time
# AND to decide whether a candidate scene's patch-area passes muster.
QUALITY_BANDS = (
    ("cloud", "max_cloud"),
    ("shadow", "max_shadow"),
    ("light_haze", "max_light_haze"),
    ("heavy_haze", "max_heavy_haze"),
    ("snow", "max_snow"),
    ("unusable", "max_unusable"),
)


def _is_bad(q: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Return (is_bad, list of failing reasons) for a UDM2 quality dict."""
    reasons: list[str] = []
    if q.get("clear", 1.0) < args.min_clear:
        reasons.append(f"clear<{args.min_clear}")
    for band, attr in QUALITY_BANDS:
        ceiling = getattr(args, attr)
        v = q.get(band, 0.0)
        if v > ceiling:
            reasons.append(f"{band}>{ceiling}")
    return (len(reasons) > 0, reasons)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_bad_patches(args: argparse.Namespace) -> list[dict]:
    """Filter UDM2 quality rows to patches failing ANY quality band."""
    rows = _read_jsonl(args.planet_root / "_global" / "udm2_quality.jsonl")
    bad: list[dict] = []
    for r in rows:
        if "clear" not in r:
            continue
        is_bad, reasons = _is_bad(r, args)
        if is_bad:
            r["_reasons"] = reasons
            bad.append(r)
    return bad


def _index_search_results(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    """Map (country, id, window) -> latest search-result row (item_id, ftw_date)."""
    by_key: dict[tuple[str, str, str], dict] = {}
    for shard in sorted((planet_root / "_global" / "search").glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") != "found":
                continue
            by_key[(r["country"], r["id"], r["window"])] = r
    return by_key


def _index_manifest_geom(planet_root: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for r in _read_jsonl(planet_root / "_global" / "manifest.jsonl"):
        out[(r["country"], r["id"], r["window"])] = r
    return out


async def _search_candidates(
    sess: Session,
    geom: dict,
    target_date: datetime,
    search_days: int,
    scene_max_cloud: float,
    exclude_item_id: str,
    limit: int = 50,
) -> list[dict]:
    """Return up to ``limit`` candidate scenes, ordered by score."""
    from typing import Any as _Any

    cl: _Any = sess.client("data")
    gte = (target_date - timedelta(days=search_days)).replace(tzinfo=UTC)
    lte = (target_date + timedelta(days=search_days)).replace(tzinfo=UTC)
    sf = data_filter.and_filter(
        [
            data_filter.geometry_filter(geom),
            data_filter.date_range_filter("acquired", gte=gte, lte=lte),
            data_filter.range_filter("cloud_cover", lte=scene_max_cloud),
            data_filter.permission_filter(),
            data_filter.std_quality_filter(),
        ]
    )

    pgeom = shape(geom)
    parea = max(pgeom.area, 1e-12)

    cands: list[dict] = []
    async for it in cl.search([ITEM_TYPE], search_filter=sf, limit=200):
        if it["id"] == exclude_item_id:
            continue
        try:
            scene_geom = shape(it["geometry"])
        except (KeyError, ValueError):
            continue
        coverage = pgeom.intersection(scene_geom).area / parea
        if coverage < 0.99:
            continue
        props = it["properties"]
        cands.append(
            {
                "item_id": it["id"],
                "acquired": datetime.fromisoformat(props["acquired"]),
                "cloud_cover": float(props.get("cloud_cover", 1.0)),
                "coverage": coverage,
            }
        )
        if len(cands) >= limit:
            break

    target_utc = target_date.replace(tzinfo=UTC) if target_date.tzinfo is None else target_date

    def _score(c: dict) -> tuple:
        return (abs((c["acquired"] - target_utc).total_seconds()), c["cloud_cover"])

    cands.sort(key=_score)
    return cands


def _patch_quality_from_udm2(
    udm2_url: str,
    aoi_4326: dict,
) -> dict:
    """Range-read UDM2 over the patch AOI and compute per-band fractions."""
    with cog_env(), rasterio.open(udm2_url) as src:
        src_crs = src.crs.to_string()
        bounds = aoi_bounds_in_target(aoi_4326, src_crs)
        import math

        w = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        col = math.floor(w.col_off)
        row = math.floor(w.row_off)
        width = math.ceil(w.col_off + w.width) - col
        height = math.ceil(w.row_off + w.height) - row
        window = rasterio.windows.Window.from_slices(
            (row, row + height), (col, col + width), boundless=True
        )
        data = src.read(window=window)
    n = data.shape[1] * data.shape[2]
    return {
        "clear": float((data[0] > 0).sum()) / n,
        "snow": float((data[1] > 0).sum()) / n,
        "shadow": float((data[2] > 0).sum()) / n,
        "light_haze": float((data[3] > 0).sum()) / n,
        "heavy_haze": float((data[4] > 0).sum()) / n,
        "cloud": float((data[5] > 0).sum()) / n,
        "unusable": float((data[7] > 0).sum()) / n,
    }


async def _process_patch(
    sess: Session,
    bad: dict,
    search_row: dict,
    geom: dict,
    args: argparse.Namespace,
    scene_udm2_cache: dict[str, str],
) -> dict:
    """Find a better scene for this patch and re-extract."""
    country = bad["country"]
    pid = bad["id"]
    window = bad["window"]
    out_dir = args.planet_root / country
    out_dir / f"{pid}_{window}.tif"
    out_dir / f"{pid}_{window}_udm2.tif"

    target_date = datetime.fromisoformat(search_row["scene_date"])
    # Determine search_days: ±60d default covers most FTW season windows.
    search_days = args.search_days if args.search_days > 0 else 60

    cands = await _search_candidates(
        sess,
        geom,
        target_date,
        search_days,
        args.scene_max_cloud,
        exclude_item_id=search_row["item_id"],
        limit=50,
    )
    base = {
        "country": country,
        "id": pid,
        "window": window,
        "previous_item_id": search_row["item_id"],
        "previous_clear": bad.get("clear"),
        "previous_cloud": bad.get("cloud"),
        "previous_shadow": bad.get("shadow"),
        "previous_unusable": bad.get("unusable"),
        "previous_reasons": bad.get("_reasons", []),
        "n_candidates": len(cands),
    }
    if not cands:
        return {**base, "status": "no_candidate"}

    probed: list[dict] = []  # remember probe results so we can pick the best fallback
    for cand in cands[: args.max_candidates]:
        iid = cand["item_id"]
        try:
            url = scene_udm2_cache.get(iid)
            if url is None:
                url = await activate_asset_url(sess, iid, ASSET_UDM2)
                scene_udm2_cache[iid] = url
            patch_q = await asyncio.to_thread(_patch_quality_from_udm2, url, geom)
        except Exception as e:
            log.warning("UDM2 probe failed for %s: %s", iid, e)
            continue
        probed.append(
            {
                "iid": iid,
                "udm2_url": url,
                "patch_q": patch_q,
                "scene_date": cand["acquired"],
                "scene_cc": cand["cloud_cover"],
            }
        )
        is_bad_cand, _ = _is_bad(patch_q, args)
        if not is_bad_cand:
            # Found a fully passing scene. Activate SR + re-extract.
            try:
                sr_url = await activate_asset_url(sess, iid, ASSET_SR)
                udm2_url = url  # already activated
                await asyncio.to_thread(
                    _extract_scene_group,
                    iid,
                    sr_url,
                    udm2_url,
                    [{"id": pid, "window": window, "geometry_4326": geom}],
                    out_dir,
                )
                # Force overwrite by removing old file before re-extracting.
                # Actually _extract_scene_group has skipped_existing logic — delete first.
                # Already deleted above? No, wasn't. Need to delete first then rerun.
            except Exception as e:
                return {
                    **base,
                    "status": "extract_failed",
                    "error": str(e),
                    "candidate": iid,
                    "candidate_clear": patch_q["clear"],
                }
            return {
                **base,
                "status": "matched_new",
                "new_item_id": iid,
                "new_scene_date": cand["acquired"].isoformat(),
                "new_clear": round(patch_q["clear"], 4),
                "new_unusable": round(patch_q["unusable"], 4),
            }

    # No candidate fully passes — fall back to the *best probed* if it strictly
    # improves on the original's clear%. Otherwise give up.
    if probed:
        best = max(probed, key=lambda x: x["patch_q"]["clear"])
        prev_clear = bad.get("clear", 0.0)
        if best["patch_q"]["clear"] > prev_clear + 0.05:
            iid = best["iid"]
            try:
                sr_url = await activate_asset_url(sess, iid, ASSET_SR)
                udm2_url = best["udm2_url"]
                _ = await asyncio.to_thread(
                    _extract_scene_group,
                    iid,
                    sr_url,
                    udm2_url,
                    [{"id": pid, "window": window, "geometry_4326": geom}],
                    out_dir,
                )
            except Exception as e:
                return {**base, "status": "extract_failed", "error": str(e), "candidate": iid}
            return {
                **base,
                "status": "matched_better_partial",
                "new_item_id": iid,
                "new_scene_date": best["scene_date"].isoformat(),
                "new_clear": round(best["patch_q"]["clear"], 4),
                "new_unusable": round(best["patch_q"]["unusable"], 4),
            }
    return {**base, "status": "no_better_candidate"}


async def _run(args: argparse.Namespace) -> None:
    require_api_key()
    bad = _load_bad_patches(args)
    log.info("bad patches to consider: %d", len(bad))

    log_path = args.planet_root / "_global" / "resample_log.jsonl"
    done = {(r["country"], r["id"], r["window"]) for r in _read_jsonl(log_path)}
    log.info("already processed: %d", len(done))

    todo = [r for r in bad if (r["country"], r["id"], r["window"]) not in done]
    log.info("to process: %d", len(todo))
    if not todo:
        return

    search_idx = _index_search_results(args.planet_root)
    manifest_idx = _index_manifest_geom(args.planet_root)

    sem = asyncio.Semaphore(args.concurrency)
    scene_udm2_cache: dict[str, str] = {}

    async def _wrapped(b: dict) -> dict:
        async with sem:
            key = (b["country"], b["id"], b["window"])
            search_row = search_idx.get(key)
            man_row = manifest_idx.get(key)
            if not search_row or not man_row:
                return {
                    "country": b["country"],
                    "id": b["id"],
                    "window": b["window"],
                    "status": "missing_search_or_manifest",
                }
            geom = man_row["geometry_4326"]
            # Wipe existing SR + UDM2 so re-extraction doesn't hit skipped_existing.
            sr_path = args.planet_root / b["country"] / f"{b['id']}_{b['window']}.tif"
            udm2_path = args.planet_root / b["country"] / f"{b['id']}_{b['window']}_udm2.tif"
            sr_path.unlink(missing_ok=True)
            udm2_path.unlink(missing_ok=True)
            return await _process_patch(sess, b, search_row, geom, args, scene_udm2_cache)

    async with Session() as sess:
        coros = [_wrapped(b) for b in todo]
        with log_path.open("a") as f:
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
    for noisy in ("planet", "httpx", "httpcore", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
