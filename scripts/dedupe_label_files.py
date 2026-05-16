"""Collapse per-window label tifs into a single per-patch label.

Old convention: ``<id>_a_label.tif`` and ``<id>_b_label.tif`` (identical
because the label is per-patch — window_a and window_b share AOI + UTM grid).

New convention: a single ``<id>_label.tif``.

Migration per patch:
  1. If ``<id>_label.tif`` exists: delete any ``<id>_<w>_label.tif`` siblings.
  2. Else if ``<id>_a_label.tif`` exists: rename to ``<id>_label.tif`` and
     delete ``<id>_b_label.tif`` if present.
  3. Else if only ``<id>_b_label.tif`` exists: rename to ``<id>_label.tif``.
  4. Else: nothing to do (label missing; rasterize_labels.py will produce one).

Idempotent. Dry-run by default — pass ``--apply`` to actually move/delete.

Example:
    uv run scripts/dedupe_label_files.py --apply
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--apply", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total_renamed = 0
    total_removed = 0
    total_already = 0
    total_missing = 0

    for d in sorted(args.planet_root.iterdir()):
        if not d.is_dir() or d.name == "_global":
            continue
        country = d.name
        # Group label files by patch id.
        by_pid: dict[str, dict[str, Path]] = {}
        for tif in d.glob("*_label.tif"):
            stem = tif.stem  # <id>_<window>_label OR <id>_label
            parts = stem.split("_")
            # If second-to-last token is 'a' or 'b' followed by 'label', it's per-window.
            if len(parts) >= 3 and parts[-1] == "label" and parts[-2] in ("a", "b"):
                pid = "_".join(parts[:-2])
                win = parts[-2]
                by_pid.setdefault(pid, {})[win] = tif
            elif len(parts) >= 2 and parts[-1] == "label":
                pid = "_".join(parts[:-1])
                by_pid.setdefault(pid, {})["_"] = tif

        renamed = removed = already = missing = 0
        for pid, files in by_pid.items():
            new_path = d / f"{pid}_label.tif"
            if "_" in files:
                # Already in new form. Drop any per-window leftovers.
                already += 1
                for w in ("a", "b"):
                    if w in files:
                        if args.apply:
                            files[w].unlink()
                        removed += 1
                continue
            src = files.get("a") or files.get("b")
            if src is None:
                missing += 1
                continue
            if args.apply:
                src.rename(new_path)
            renamed += 1
            # Delete the other window's copy if present.
            other_win = "b" if "a" in files else "a"
            if other_win in files and files[other_win] != src:
                if args.apply:
                    files[other_win].unlink()
                removed += 1

        print(
            f"{country:14s} renamed={renamed:>5d} removed={removed:>5d} "
            f"already={already:>5d} missing={missing:>5d}"
        )
        total_renamed += renamed
        total_removed += removed
        total_already += already
        total_missing += missing

    print()
    print(
        f"TOTAL: renamed={total_renamed}  removed={total_removed}  "
        f"already_new={total_already}  missing={total_missing}  "
        f"{'(applied)' if args.apply else '(dry-run)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
