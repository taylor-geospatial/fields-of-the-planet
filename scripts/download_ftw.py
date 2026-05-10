"""Download one FTW country and emit a per-patch JSONL index.

Wraps the upstream `ftw` CLI for the actual download, then walks the result
to build an index that the Planet matching step consumes.

Example:
    uv run scripts/download_ftw.py --country rwanda --root data/ftw
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ftw_planet.ftw import FTWCountry, write_index


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--country",
        required=True,
        help="FTW country slug (e.g. rwanda) or 'all' to fetch every country.",
    )
    p.add_argument("--root", type=Path, default=Path("data/ftw"), help="FTW download root")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Only (re)build the index — assume data is already on disk.",
    )
    return p.parse_args()


def _index_country(root: Path, country: str) -> int:
    print(f"Indexing {root / country} ...")
    c = FTWCountry(country=country, root=root)
    patches = c.patches()
    print(f"  {len(patches)} patches")
    index_path = root / country / "index.jsonl"
    write_index(patches, index_path)
    print(f"Wrote {index_path}")
    return len(patches)


def main() -> int:
    args = parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if shutil.which("ftw") is None:
            print(
                "ERROR: `ftw` CLI not found. Run `make install` (adds ftw-tools).",
                file=sys.stderr,
            )
            return 1
        # ftw-tools puts data under <out>/ftw/<country>/...; pass parent so that
        # the final layout is data/ftw/<country>.
        out_parent = root.parent
        cmd = ["ftw", "data", "download", "--countries", args.country, "--out", str(out_parent)]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if args.country == "all":
        total = 0
        for d in sorted((root).iterdir()):
            if d.is_dir() and any(d.glob("chips_*.parquet")):
                total += _index_country(root, d.name)
        print(f"Total patches indexed across all countries: {total}")
    else:
        _index_country(root, args.country)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
