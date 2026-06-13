"""Shared helpers to macro-average per-country eval CSVs over a country set.

Canonical held-out country sets for the FTP paper:

- HELDOUT_11: the 11 held-out CC-BY countries that appear in
  ``logs/heldout/*.csv`` (verified against the CSV ``country`` column).
- HELDOUT_10_DENSE: HELDOUT_11 minus {kenya}. Kenya's labels are
  presence-only (background untrusted), so its supervised pixel/object
  metrics are not comparable; the headline macro excludes it and Kenya is
  reported separately as a presence-only stress-test row.
- HELDOUT_9: HELDOUT_11 minus {kenya, portugal}, the older diagnostic set
  (kenya + portugal are sparse / low-coverage and were excluded from earlier
  headline numbers).

Use ``HELDOUT_10_DENSE`` as the headline; ``HELDOUT_11`` and ``HELDOUT_9``
are kept so prior numbers stay reproducible.
"""

from pathlib import Path

import pandas as pd

HELDOUT_11: tuple[str, ...] = (
    "belgium",
    "cambodia",
    "croatia",
    "germany",
    "kenya",
    "latvia",
    "lithuania",
    "portugal",
    "slovenia",
    "south_africa",
    "sweden",
)

HELDOUT_10_DENSE: tuple[str, ...] = tuple(c for c in HELDOUT_11 if c != "kenya")

HELDOUT_9: tuple[str, ...] = tuple(c for c in HELDOUT_11 if c not in {"kenya", "portugal"})

# Default columns to aggregate when present.
_DEFAULT_METRICS = (
    "pixel_level_iou",
    "pixel_level_precision",
    "pixel_level_recall",
    "object_pix_f1",
    "object_ws_f1",
    "object_level_f1",
)


def _country_col(df: pd.DataFrame) -> str:
    if "country" in df.columns:
        return "country"
    if "countries" in df.columns:
        return "countries"
    raise KeyError(f"no country/countries column in CSV; have {list(df.columns)}")


def load_and_filter(csv_path: str | Path, countries: tuple[str, ...]) -> pd.DataFrame:
    """Load a per-country eval CSV and return rows for ``countries`` only.

    Does not silently drop missing countries — caller can compare
    ``len(out)`` vs ``len(countries)`` to detect gaps.
    """
    df = pd.read_csv(csv_path)
    ccol = _country_col(df)
    return df[df[ccol].isin(set(countries))].copy()


def macro_avg(
    csv_path: str | Path,
    countries: tuple[str, ...],
    metrics: tuple[str, ...] | None = None,
) -> dict[str, float | int]:
    """Macro-average ``metrics`` across ``countries`` from a per-country CSV.

    Returns a dict with one entry per metric present in the CSV, plus
    ``n_countries`` (rows actually averaged) and ``n_expected``
    (``len(countries)``). Missing countries reduce ``n_countries`` but do not
    raise — the caller is expected to flag the gap.
    """
    sub = load_and_filter(csv_path, countries)
    cols = tuple(metrics) if metrics is not None else _DEFAULT_METRICS
    out: dict[str, float | int] = {}
    for c in cols:
        if c in sub.columns:
            out[c] = float(sub[c].mean())
    out["n_countries"] = len(sub)
    out["n_expected"] = len(countries)
    return out


def aggregate_table(
    rows: list[tuple[str, str | Path]],
    countries: tuple[str, ...],
    metrics: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame of macro-averages, one row per (label, csv)."""
    records: list[dict[str, float | int | str]] = []
    for label, path in rows:
        agg: dict[str, float | int | str] = dict(macro_avg(path, countries, metrics=metrics))
        agg["label"] = label
        agg["csv"] = str(path)
        records.append(agg)
    df = pd.DataFrame.from_records(records)
    # Stable column order: label, metrics..., n, csv
    metric_cols = [c for c in df.columns if c not in {"label", "csv", "n_countries", "n_expected"}]
    return df[["label", *metric_cols, "n_countries", "n_expected", "csv"]]
