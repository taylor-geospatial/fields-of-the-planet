"""Regenerate the standalone results bundle in ``results/metrics/``.

A human-readable export of the headline polygon metrics the FTP paper reports,
on the true-GT / native-GSD protocol, traced to source. The per-method macro
reuses the exact Table 1 aggregation from ``polygon_metrics_table.py``, so the
bundle cannot drift from the paper.

Writes (metrics as fractions in [0,1]; the paper shows them x100):
  results/metrics/macro_summary.csv        per-method dense-10 macro (= Table 1)
  results/metrics/per_country_metrics.csv  per-country true-GT overall + PQ-by-size

The README is maintained by hand; rerun this script after any eval refresh.

Run::

    uv run python paper/scripts/results_bundle.py
"""

import csv
from pathlib import Path

import pandas as pd
from _aggregate import HELDOUT_10_DENSE
from polygon_metrics_table import (
    AREA_BINS,
    BND_COLS,
    POLY_COLS,
    RESABL,
    ROWS,
    _area_pq,
    _country_csvs,
    _macro,
    _norm_count,
)

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / "results" / "metrics"

# Plain method labels, in the same row order as polygon_metrics_table.ROWS
# (avoids stripping LaTeX out of the table's display names).
METHOD_LABELS = [
    "DelineateAnything (Planet)",
    "DelineateAnything-S (Planet)",
    "FTW-PRUE+ (S2)",
    "FTW-PRUE+ (S2)",
    "FTP-PRUE+ (Planet)",
    "FTP-PRUE+ (Planet)",
]

# Per-country true-GT runs for the four segmentation models (dir under RESABL).
SEG_CONDITIONS = [
    ("FTW-PRUE+ (S2)", "B3", "s2b3_10m"),
    ("FTW-PRUE+ (S2)", "B7", "s2nat10"),
    ("FTP-PRUE+ (Planet)", "B3", "planet3m"),
    ("FTP-PRUE+ (Planet)", "B7", "planetb7_3m"),
]


def write_macro_summary() -> Path:
    """Per-method macro over the 10 dense held-out countries (matches Table 1)."""
    out = OUT_DIR / "macro_summary.csv"
    fields = [
        "method",
        "backbone",
        "pq",
        "sq",
        "rq",
        "f1_5_95",
        "dN_over_N",
        "bnd_mean_m",
        "bnd_p95_m",
        "pixel_iou",
        "pq_small",
        "pq_medium",
        "pq_large",
    ]
    if len(METHOD_LABELS) != len(ROWS):
        raise RuntimeError(f"{len(METHOD_LABELS)} labels but {len(ROWS)} table rows")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, (_name, backbone, poly, area_csv, pix_csv, bnd_csv, _, _) in zip(
            METHOD_LABELS, ROWS
        ):
            agg = {c: _macro(poly, c) for c in POLY_COLS}
            agg["dN_norm"] = _norm_count(poly)
            bnd = bnd_csv if bnd_csv is not None else poly
            for c in BND_COLS:
                agg[c] = _macro(bnd, c)
            agg["pixel_iou"] = _macro(pix_csv, "pixel_level_iou")
            bins = {b: float("nan") for b in AREA_BINS} if area_csv is None else _area_pq(area_csv)

            def r(v: float, nd: int = 4) -> str:
                return "" if v != v else f"{v:.{nd}f}"

            w.writerow(
                {
                    "method": label,
                    "backbone": backbone,
                    "pq": r(agg["pq"]),
                    "sq": r(agg["pq_sq"]),
                    "rq": r(agg["pq_rq"]),
                    "f1_5_95": r(agg["ap_5_95"]),
                    "dN_over_N": r(agg["dN_norm"]),
                    "bnd_mean_m": r(agg["boundary_error_m_mean"], 2),
                    "bnd_p95_m": r(agg["boundary_error_m_p95"], 2),
                    "pixel_iou": r(agg["pixel_iou"]),
                    "pq_small": r(bins["small"]),
                    "pq_medium": r(bins["medium"]),
                    "pq_large": r(bins["large"]),
                }
            )
    return out


def write_per_country() -> Path:
    """Per-country true-GT overall + PQ-by-size for the four segmentation models."""
    out = OUT_DIR / "per_country_metrics.csv"
    # Boundary chamfer is omitted here: the true-GT native-GSD runs skip it (it is
    # a macro-only quantity in the paper, sourced from a separate run). See
    # macro_summary.csv for the macro boundary error.
    fields = [
        "method",
        "backbone",
        "condition",
        "country",
        "n_patches",
        "pq",
        "sq",
        "rq",
        "f1_5_95",
        "n_pred_mean",
        "n_gt_mean",
        "dN_over_N",
        "pq_small",
        "pq_medium",
        "pq_large",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, backbone, cond in SEG_CONDITIONS:
            for cpath in _country_csvs(RESABL / cond):
                d = pd.read_csv(cpath).iloc[0]
                b = pd.read_csv(f"{cpath}.bins.csv").set_index("bin")
                n_gt = float(d["n_gt_mean"])
                w.writerow(
                    {
                        "method": method,
                        "backbone": backbone,
                        "condition": cond,
                        "country": d["country"],
                        "n_patches": int(d["n_patches"]),
                        "pq": f"{d['pq']:.4f}",
                        "sq": f"{d['pq_sq']:.4f}",
                        "rq": f"{d['pq_rq']:.4f}",
                        "f1_5_95": f"{d['ap_5_95']:.4f}",
                        "n_pred_mean": f"{d['n_pred_mean']:.2f}",
                        "n_gt_mean": f"{n_gt:.2f}",
                        "dN_over_N": f"{abs(d['n_pred_mean'] - n_gt) / n_gt:.4f}",
                        "pq_small": f"{b.loc['small', 'pq']:.4f}",
                        "pq_medium": f"{b.loc['medium', 'pq']:.4f}",
                        "pq_large": f"{b.loc['large', 'pq']:.4f}",
                    }
                )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (write_macro_summary(), write_per_country()):
        print(f"wrote {p}")
    print(f"  dense-10 held-out: {', '.join(HELDOUT_10_DENSE)}")


if __name__ == "__main__":
    main()
