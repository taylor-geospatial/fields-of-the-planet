"""Download FTW field-boundary GeoParquet polygons from Source Cooperative.

For each FTW country we know about, hit the corresponding source.coop
``data.source.coop/kerner-lab/fields-of-the-world-<repo>/boundaries_*.parquet``
URL and save to ``data/ftw_polygons/<country>.parquet``.

Country -> (repo_slug, filename) mapping was discovered by inspection of
each per-country source.coop repo. Year suffixes vary; some countries use
multi-year filenames (germany ``2018_2019``).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SOURCES: dict[str, tuple[str, str]] = {
    "austria": ("fields-of-the-world-austria", "boundaries_austria_2021.parquet"),
    "belgium": ("fields-of-the-world-belgium", "boundaries_belgium_2021.parquet"),
    "brazil": ("fields-of-the-world-brazil", "boundaries_brazil_2020.parquet"),
    "cambodia": ("fields-of-the-world-cambodia", "boundaries_cambodia_2021.parquet"),
    "corsica": ("fields-of-the-world-corsica", "boundaries_corsica_2021.parquet"),
    "croatia": ("fields-of-the-world-croatia", "boundaries_croatia_2023.parquet"),
    "denmark": ("fields-of-the-world-denmark", "boundaries_denmark_2021.parquet"),
    "estonia": ("fields-of-the-world-estonia", "boundaries_estonia_2021.parquet"),
    "finland": ("fields-of-the-world-finland", "boundaries_finland_2021.parquet"),
    "france": ("fields-of-the-world-france", "boundaries_france_2020.parquet"),
    "germany": ("fields-of-the-world-germany", "boundaries_germany_2018_2019.parquet"),
    "india": ("fields-of-the-world-india", "boundaries_india_2016.parquet"),
    "kenya": ("fields-of-the-world-kenya", "boundaries_kenya_2022.parquet"),
    "latvia": ("fields-of-the-world-latvia", "boundaries_latvia_2021.parquet"),
    "lithuania": ("fields-of-the-world-lithuania", "boundaries_lithuania_2021.parquet"),
    "luxembourg": ("fields-of-the-world-luxembourg", "boundaries_luxembourg_2022.parquet"),
    "netherlands": ("fields-of-the-world-netherlands", "boundaries_netherlands_2022.parquet"),
    "portugal": ("fields-of-the-world-portugal", "boundaries_portugal_2021.parquet"),
    "rwanda": ("fields-of-the-world-rwanda", "boundaries_rwanda_2021.parquet"),
    "slovakia": ("fields-of-the-world-slovakia", "boundaries_slovakia_2021.parquet"),
    "slovenia": ("fields-of-the-world-slovenia", "boundaries_slovenia_2021.parquet"),
    "south_africa": ("fields-of-the-world-southafrica", "boundaries_south_africa_2018.parquet"),
    "spain": ("fields-of-the-world-spain", "boundaries_spain_2020.parquet"),
    "sweden": ("fields-of-the-world-sweden", "boundaries_sweden_2021.parquet"),
    "vietnam": ("fields-of-the-world-vietnam", "boundaries_vietnam_2021.parquet"),
}

BASE_URL = "https://data.source.coop/kerner-lab"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/ftw_polygons"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--country", default="all", help="Single country slug or 'all'.")
    return p.parse_args()


def _download(country: str, repo: str, fname: str, out_dir: Path) -> tuple[str, str, int]:
    out_path = out_dir / f"{country}.parquet"
    if out_path.exists():
        return country, "skip-existing", out_path.stat().st_size
    url = f"{BASE_URL}/{repo}/{fname}"
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    tmp = out_path.with_suffix(".parquet.part")
    n = 0
    with tmp.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                n += len(chunk)
    tmp.rename(out_path)
    return country, "ok", n


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    selected = SOURCES if args.country == "all" else {args.country: SOURCES[args.country]}
    print(f"Downloading {len(selected)} country polygon files to {args.out}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(_download, c, repo, fname, args.out) for c, (repo, fname) in selected.items()
        ]
        for fut in futs:
            country, status, size = fut.result()
            print(f"  {country:18s} {status:14s} {size / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
