"""Merge 11-country held-out polygon_metrics CSVs with the 13 newly computed
FTW 22-country test rows. Writes ``*_22.csv`` alongside the originals and
prints macro PQ across the 22 countries.

The FTW full test split (22 countries) is the one from
``logs/ftw_official/b7_per_country.csv``. The existing 11-country CSVs
include ``kenya`` which is NOT in the 22-country list — so the merged CSV
filters to the canonical 22.
"""

import argparse
from pathlib import Path

import pandas as pd

FTW22 = [
    "austria",
    "belgium",
    "brazil",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "portugal",
    "rwanda",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]


def merge(existing: Path, new: Path, out: Path) -> pd.DataFrame:
    df_a = pd.read_csv(existing)
    df_b = pd.read_csv(new)
    merged = pd.concat([df_a, df_b], ignore_index=True)
    # Drop duplicates by country (prefer new if any overlap).
    merged = merged.drop_duplicates(subset=["country"], keep="last")
    # Filter to canonical 22.
    merged = merged[merged["country"].isin(FTW22)].copy()
    merged = merged.sort_values("country").reset_index(drop=True)
    merged.to_csv(out, index=False)
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--planet-existing",
        type=Path,
        default=Path("logs/polygon_metrics/planet_b3_augmax_full.csv"),
    )
    p.add_argument(
        "--planet-new",
        type=Path,
        default=Path("logs/polygon_metrics/planet_b3_augmax_full_missing13.csv"),
    )
    p.add_argument(
        "--planet-out", type=Path, default=Path("logs/polygon_metrics/planet_b3_augmax_full_22.csv")
    )
    p.add_argument(
        "--s2-existing", type=Path, default=Path("logs/polygon_metrics/s2_b7_augmax_full.csv")
    )
    p.add_argument(
        "--s2-new", type=Path, default=Path("logs/polygon_metrics/s2_b7_augmax_full_missing13.csv")
    )
    p.add_argument(
        "--s2-out", type=Path, default=Path("logs/polygon_metrics/s2_b7_augmax_full_22.csv")
    )
    args = p.parse_args()

    for label, exist, new, out in [
        ("planet", args.planet_existing, args.planet_new, args.planet_out),
        ("s2", args.s2_existing, args.s2_new, args.s2_out),
    ]:
        merged = merge(exist, new, out)
        n = len(merged)
        macro_pq = merged["pq"].mean()
        macro_sq = merged["pq_sq"].mean()
        macro_rq = merged["pq_rq"].mean()
        macro_ap = merged["ap_5_95"].mean()
        print(f"[{label}] countries={n} out={out}")
        print(f"    macro PQ={macro_pq:.4f} SQ={macro_sq:.4f} RQ={macro_rq:.4f} AP={macro_ap:.4f}")
        missing = sorted(set(FTW22) - set(merged["country"].tolist()))
        if missing:
            print(f"    MISSING from 22: {missing}")


if __name__ == "__main__":
    main()
