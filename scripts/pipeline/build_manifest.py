"""Phase 0 — combine every FTW country's index.jsonl into one global manifest.

Output: ``<out>/_global/manifest.jsonl``. One row per (patch, window) so
downstream shards can be sliced uniformly without caring about country
boundaries.

Example:
    uv run scripts/build_manifest.py --ftw-root data/ftw --out data/planet
"""

import argparse
import json
from pathlib import Path

from ftw_planet.ftw import read_index

WINDOWS = ("a", "b")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ftw-root", type=Path, default=Path("data/ftw"))
    p.add_argument("--out", type=Path, default=Path("data/planet"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out / "_global"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    countries = sorted(
        d.name for d in args.ftw_root.iterdir() if d.is_dir() and (d / "index.jsonl").exists()
    )
    print(f"Found {len(countries)} indexed countries: {', '.join(countries)}")

    n_rows = 0
    with manifest_path.open("w") as f:
        for c in countries:
            patches = read_index(args.ftw_root / c / "index.jsonl")
            for p in patches:
                for w in WINDOWS:
                    row = {
                        "country": c,
                        "id": p["id"],
                        "window": w,
                        "ftw_date": p[f"win_{w}_date"],
                        "range": p[f"win_{w}_range"],
                        "geometry_4326": p["geometry_4326"],
                    }
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
    print(f"Wrote {n_rows} rows to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
