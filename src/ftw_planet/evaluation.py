"""Country sets and macro-averaging for Fields of the Planet (FTP) evaluation.

Kenya's FTW labels are presence-only: the tile annotates a subset of fields
and leaves surrounding land unlabeled, so its background class is untrusted
and supervised pixel/object metrics against it are not comparable across
models. Headline macro-averages therefore use the dense-label countries and
report Kenya separately.
"""

from pathlib import Path

import pandas as pd

HELDOUT_COUNTRIES = (
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

PRESENCE_ONLY_COUNTRIES = ("kenya",)

DENSE_LABEL_COUNTRIES = tuple(c for c in HELDOUT_COUNTRIES if c not in PRESENCE_ONLY_COUNTRIES)

FULLDATA_REGIONS = (
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
)

DEFAULT_METRICS = (
    "pixel_level_iou",
    "object_ws_f1",
    "pq",
    "pq_sq",
    "pq_rq",
    "ap_5_95",
    "polygon_count_delta_mean",
    "boundary_error_m_mean",
)


def country_column(df: pd.DataFrame) -> str:
    for name in ("country", "countries"):
        if name in df.columns:
            return name
    raise KeyError(f"no country column in CSV; columns are {list(df.columns)}")


def load_country_rows(csv_path: str | Path, countries: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col = country_column(df)
    return df[df[col].isin(set(countries))].copy()


def macro_average(
    csv_path: str | Path,
    countries: tuple[str, ...],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Macro-average each metric over the rows matching ``countries``.

    Missing countries lower the row count rather than raising; the caller can
    compare ``n_countries`` against ``n_expected`` to detect gaps.
    """
    rows = load_country_rows(csv_path, countries)
    result: dict[str, float] = {
        metric: float(rows[metric].mean(skipna=True))
        for metric in metrics
        if metric in rows.columns
    }
    result["n_countries"] = len(rows)
    result["n_expected"] = len(countries)
    return result
