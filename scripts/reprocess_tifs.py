"""Reprocess every GeoTIFF under ``data/planet/<country>/`` in place.

Rewrites each tif with maximum-compression ZSTD (level 22), a single-strip
(non-tiled) layout suited to whole-patch reads, drops overviews, and
embeds rich provenance tags pulled from the pipeline's JSONL logs.

Per-asset settings:

* uint16 SR (``*_a.tif`` / ``*_b.tif`` excluding ``_udm2`` and ``_label``):
  ``predictor=2`` (horizontal differencing).
* uint8 UDM2 (``*_udm2.tif``): ``predictor=2``.
* uint8 labels (``*_label.tif``): ``predictor=1`` (categorical).

Idempotent: skips any tif that already advertises ``PROCESSING_VERSION``.

Example:
    uv run scripts/reprocess_tifs.py --country rwanda --dry-run --workers 4
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import rasterio
from rich.logging import RichHandler

log = logging.getLogger("ftw_planet.reprocess")

PROCESSING_VERSION = "ftw-planet v1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--country", default="all", help="FTW country slug, or 'all'.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write; just report planned work and projected size.",
    )
    p.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help="Override per-file JSONL log path (default: <planet-root>/_global/reprocess_log.jsonl).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSONL log indexing
# ---------------------------------------------------------------------------


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


def _index_provenance(
    planet_root: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Map (country, id, window) -> {item_id, source}.

    Last-write-wins precedence (later overrides earlier):
        extract shards (source=original)
        resample_log (source=resampled)
        restore_log (source=restored)
    """
    g = planet_root / "_global"
    out: dict[tuple[str, str, str], dict[str, Any]] = {}

    # extract shards
    for shard in sorted((g / "extract").glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            if r.get("status") not in ("matched", "skipped_existing"):
                continue
            iid = r.get("item_id")
            if not iid:
                continue
            k = (str(r.get("country", "")), str(r.get("id", "")), str(r.get("window", "")))
            out[k] = {"item_id": str(iid), "source": "original"}

    # resample_log: prefer new_item_id, else item_id
    for r in _read_jsonl(g / "resample_log.jsonl"):
        status = r.get("status_resample") or r.get("status")
        if status not in ("matched", "matched_new", "skipped_existing"):
            continue
        iid = r.get("new_item_id") or r.get("item_id")
        if not iid:
            continue
        k = (str(r.get("country", "")), str(r.get("id", "")), str(r.get("window", "")))
        out[k] = {"item_id": str(iid), "source": "resampled"}

    # restore_log
    for r in _read_jsonl(g / "restore_log.jsonl"):
        status = r.get("status_restore") or r.get("status")
        if status not in ("matched", "restored", "skipped_existing"):
            continue
        iid = r.get("item_id") or r.get("new_item_id")
        if not iid:
            continue
        k = (str(r.get("country", "")), str(r.get("id", "")), str(r.get("window", "")))
        out[k] = {"item_id": str(iid), "source": "restored"}

    return out


def _index_search(planet_root: Path) -> dict[str, dict]:
    """Map item_id -> {scene_date, cloud_cover, coverage} from search shards."""
    out: dict[str, dict] = {}
    for shard in sorted((planet_root / "_global" / "search").glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            iid = r.get("item_id")
            if not iid or r.get("status") != "found":
                continue
            out[str(iid)] = {
                "scene_date": r.get("scene_date"),
                "cloud_cover": r.get("cloud_cover"),
                "coverage": r.get("coverage"),
            }
    return out


def _index_resample_scene(planet_root: Path) -> dict[str, dict]:
    """Map resampled new_item_id -> scene_date (search shards don't cover them)."""
    out: dict[str, dict] = {}
    for r in _read_jsonl(planet_root / "_global" / "resample_log.jsonl"):
        iid = r.get("new_item_id")
        if not iid:
            continue
        out[str(iid)] = {"scene_date": r.get("new_scene_date")}
    return out


def _index_manifest(
    planet_root: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in _read_jsonl(planet_root / "_global" / "manifest.jsonl"):
        k = (str(r.get("country", "")), str(r.get("id", "")), str(r.get("window", "")))
        rng = r.get("range") or [None, None]
        out[k] = {
            "ftw_target_date": r.get("ftw_date"),
            "season_start": rng[0] if len(rng) > 0 else None,
            "season_end": rng[1] if len(rng) > 1 else None,
        }
    return out


# ---------------------------------------------------------------------------
# Per-file reprocess
# ---------------------------------------------------------------------------


def _classify(path: Path) -> tuple[str, str, str] | None:
    """Return (patch_id, window, asset) or None if not a recognized tif."""
    name = path.name
    if not name.endswith(".tif"):
        return None
    stem = path.stem
    if stem.endswith("_label"):
        base = stem[: -len("_label")]
        pid, _, win = base.rpartition("_")
        if win not in ("a", "b"):
            return None
        return pid, win, "LABEL"
    if stem.endswith("_udm2"):
        base = stem[: -len("_udm2")]
        pid, _, win = base.rpartition("_")
        if win not in ("a", "b"):
            return None
        return pid, win, "UDM2"
    pid, _, win = stem.rpartition("_")
    if win not in ("a", "b"):
        return None
    return pid, win, "SR"


def _build_tags(
    country: str,
    pid: str,
    window: str,
    asset: str,
    provenance: dict[str, Any] | None,
    search_meta: dict | None,
    manifest_meta: dict | None,
) -> dict[str, str]:
    tags: dict[str, str] = {
        "COUNTRY": country,
        "PATCH_ID": pid,
        "WINDOW": window,
        "ASSET": asset,
        "PROCESSING_VERSION": PROCESSING_VERSION,
    }
    if asset == "LABEL":
        return tags

    if provenance:
        tags["ITEM_ID"] = str(provenance.get("item_id", ""))
        tags["SOURCE"] = str(provenance.get("source", ""))
    if search_meta:
        if search_meta.get("scene_date") is not None:
            tags["SCENE_DATE"] = str(search_meta["scene_date"])
        if search_meta.get("cloud_cover") is not None:
            tags["CLOUD_COVER"] = str(search_meta["cloud_cover"])
        if search_meta.get("coverage") is not None:
            tags["COVERAGE"] = str(search_meta["coverage"])
    if manifest_meta:
        if manifest_meta.get("ftw_target_date"):
            tags["FTW_TARGET_DATE"] = str(manifest_meta["ftw_target_date"])
        if manifest_meta.get("season_start"):
            tags["FTW_SEASON_START"] = str(manifest_meta["season_start"])
        if manifest_meta.get("season_end"):
            tags["FTW_SEASON_END"] = str(manifest_meta["season_end"])
    return tags


def _reprocess_one(
    path_str: str,
    country: str,
    tags: dict[str, str],
    asset: str,
    dry_run: bool,
) -> dict[str, Any]:
    path = Path(path_str)
    t0 = time.perf_counter()
    try:
        size_before = path.stat().st_size
        with rasterio.open(path) as src:
            existing = dict(src.tags() or {})
            if existing.get("PROCESSING_VERSION") == PROCESSING_VERSION and not dry_run:
                return {
                    "path": str(path),
                    "country": country,
                    "asset": asset,
                    "status": "skipped_already_done",
                    "size_before": size_before,
                    "size_after": size_before,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }
            profile = dict(src.profile)
            height = src.height
            width = src.width
            data = src.read()
            descriptions = src.descriptions
            nodata = src.nodata
            crs = src.crs
            transform = src.transform
            count = src.count

        predictor = 1 if asset == "LABEL" else 2
        profile.update(
            {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": count,
                "crs": crs,
                "transform": transform,
                "compress": "ZSTD",
                "zstd_level": 22,
                "predictor": predictor,
                "tiled": False,
                "blockysize": height,
                "interleave": "pixel",
            }
        )
        # rasterio refuses blockxsize when not tiled; strip it.
        profile.pop("blockxsize", None)
        if nodata is not None:
            profile["nodata"] = nodata

        if dry_run:
            return {
                "path": str(path),
                "country": country,
                "asset": asset,
                "status": "dry_run",
                "size_before": size_before,
                "tags": tags,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }

        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data)
                if descriptions and any(descriptions):
                    dst.descriptions = descriptions
                dst.update_tags(**tags)
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        size_after = path.stat().st_size
        return {
            "path": str(path),
            "country": country,
            "asset": asset,
            "status": "ok",
            "size_before": size_before,
            "size_after": size_after,
            "ratio": round(size_after / size_before, 4) if size_before else None,
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        return {
            "path": str(path),
            "country": country,
            "asset": asset,
            "status": "failed",
            "error": str(e),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _iter_tifs(planet_root: Path, country: str) -> list[Path]:
    if country == "all":
        dirs = sorted(d for d in planet_root.iterdir() if d.is_dir() and d.name != "_global")
    else:
        d = planet_root / country
        dirs = [d] if d.is_dir() else []
    tifs: list[Path] = []
    for d in dirs:
        tifs.extend(sorted(p for p in d.iterdir() if p.name.endswith(".tif")))
    return tifs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=True)],
    )
    for noisy in ("rasterio",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()

    log.info("indexing provenance from JSONL logs ...")
    prov = _index_provenance(args.planet_root)
    search = _index_search(args.planet_root)
    resample_scene = _index_resample_scene(args.planet_root)
    manifest = _index_manifest(args.planet_root)
    log.info(
        "indexed: %d patch->item_id, %d search rows, %d resample scenes, %d manifest rows",
        len(prov),
        len(search),
        len(resample_scene),
        len(manifest),
    )

    tifs = _iter_tifs(args.planet_root, args.country)
    log.info("found %d tifs (country=%s)", len(tifs), args.country)
    if not tifs:
        return 0

    # Build per-tif task list (path, country, tags, asset).
    tasks: list[tuple[str, str, dict[str, str], str]] = []
    for tif in tifs:
        country = tif.parent.name
        cls = _classify(tif)
        if cls is None:
            continue
        pid, win, asset = cls
        key = (country, pid, win)
        provenance = prov.get(key)
        search_meta: dict | None = None
        if provenance and provenance.get("item_id"):
            iid = provenance["item_id"]
            search_meta = dict(search.get(iid) or {})
            # resampled scenes won't appear in search shards; fall back.
            if not search_meta.get("scene_date") and iid in resample_scene:
                search_meta["scene_date"] = resample_scene[iid].get("scene_date")
        manifest_meta = manifest.get(key)
        tags = _build_tags(country, pid, win, asset, provenance, search_meta, manifest_meta)
        tasks.append((str(tif), country, tags, asset))

    log.info("dispatching %d reprocess tasks across %d workers", len(tasks), args.workers)

    log_path = args.log_out or (args.planet_root / "_global" / "reprocess_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    total_before = 0
    total_after = 0
    done = 0

    with (
        ProcessPoolExecutor(max_workers=args.workers) as ex,
        log_path.open("a") as logf,
    ):
        futs = {ex.submit(_reprocess_one, p, c, t, a, args.dry_run): p for (p, c, t, a) in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            logf.write(json.dumps(row) + "\n")
            stats[row["status"]] = stats.get(row["status"], 0) + 1
            if row.get("size_before"):
                total_before += row["size_before"]
            if row.get("size_after"):
                total_after += row["size_after"]
            done += 1
            if done % 500 == 0:
                log.info("  %d/%d done; statuses=%s", done, len(tasks), stats)

    log.info("statuses: %s", stats)
    if total_before:
        if args.dry_run:
            log.info("dry-run: %.2f GiB of input scanned", total_before / 2**30)
        else:
            log.info(
                "size: before=%.2f GiB after=%.2f GiB ratio=%.3f",
                total_before / 2**30,
                total_after / 2**30,
                total_after / total_before if total_before else 0,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
