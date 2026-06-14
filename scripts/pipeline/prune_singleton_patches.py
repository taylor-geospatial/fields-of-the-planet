"""Delete patches that only have one of the two seasonal windows.

FTW's training format expects both ``window_a`` and ``window_b`` per patch.
Singleton patches (only one window present after the Planet match) aren't
usable for the paired-window task and contribute noise to evaluation. This
script walks each country dir and removes:

  * ``<id>_a.tif``  + ``<id>_a_udm2.tif`` + ``<id>_a_label.tif``
    when the sibling ``<id>_b.tif`` is missing.
  * mirror for ``<id>_b*`` when ``<id>_a.tif`` is missing.

Dry-run by default; pass ``--apply`` to actually delete.

Example:
    uv run scripts/prune_singleton_patches.py --apply
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planet-root", type=Path, default=Path("data/planet"))
    p.add_argument("--apply", action="store_true", help="Actually delete files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total_singletons = 0
    total_files_to_remove = 0

    for d in sorted(args.planet_root.iterdir()):
        if not d.is_dir() or d.name == "_global":
            continue
        country = d.name
        # Build {pid: {windows present in SR}}
        by_pid: dict[str, set[str]] = {}
        for tif in d.iterdir():
            if not tif.name.endswith((".tif", ".tiff")):
                continue
            if "_udm2" in tif.name or "_label" in tif.name:
                continue
            stem = tif.stem  # <id>_<a|b>
            pid, _, win = stem.rpartition("_")
            if win in ("a", "b") and pid:
                by_pid.setdefault(pid, set()).add(win)

        singletons = [pid for pid, ws in by_pid.items() if ws != {"a", "b"}]
        if not singletons:
            continue
        files_for_this = []
        for pid in singletons:
            for w in by_pid[pid]:
                files_for_this += [
                    d / f"{pid}_{w}.tif",
                    d / f"{pid}_{w}_udm2.tif",
                    d / f"{pid}_{w}_label.tif",
                ]
        files_for_this = [p for p in files_for_this if p.exists()]
        print(
            f"{country:14s} {len(singletons):>5d} singleton patches, "
            f"{len(files_for_this):>5d} files to remove"
        )
        total_singletons += len(singletons)
        total_files_to_remove += len(files_for_this)
        if args.apply:
            for p in files_for_this:
                p.unlink()

    print()
    print(
        f"TOTAL: {total_singletons} singleton patches  "
        f"{total_files_to_remove} files {'removed' if args.apply else '(dry-run)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
