"""Generate ``paper/figs/polygon_metrics.tex`` (``tab:polygon_metrics``).

11-country held-out macro-average of panoptic-quality, AP, polygon-count
delta, and meter-scale boundary-error metrics for each (imagery, backbone,
train-split) configuration. Sources: ``logs/polygon_metrics/<stem>.csv``.

Boundary-error rows for kenya are NaN in the source CSVs (FTW Kenya tile
ground truth is partial); ``pandas`` ``mean()`` skips them by default and we
do the same here. All other metrics include all 11 countries.

Run::

    uv run python paper/scripts/make_polygon_metrics.py
"""

from pathlib import Path

import pandas as pd

from _aggregate import HELDOUT_11, load_and_filter

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "polygon_metrics.tex"
SRC = REPO / "logs" / "polygon_metrics"

ROWS = [
    ("S2", r"\emph{augmax} B3 (CC-BY)", "s2_b3_augmax_ccby.csv", False),
    ("S2", r"\emph{augmax} B3 (full)", "s2_b3_augmax_full.csv", False),
    ("S2", r"\emph{augmax} B7 (CC-BY)", "s2_b7_augmax_ccby.csv", False),
    ("S2", r"\emph{augmax} B7 (full)", "s2_b7_augmax_full.csv", False),
    ("Planet", r"\emph{augmax} B3 (CC-BY)", "planet_b3_augmax_ccby.csv", False),
    ("Planet", r"\emph{augmax} B7 (CC-BY)", "planet_b7_augmax_ccby.csv", False),
    ("Planet", r"\textbf{\emph{augmax} B3 (full)}", "planet_b3_augmax_full.csv", True),
]

COLS = (
    "pq",
    "pq_sq",
    "pq_rq",
    "ap_5_95",
    "polygon_count_delta_mean",
    "boundary_error_m_mean",
    "boundary_error_m_p95",
)


def main() -> None:
    aggregates: list[dict[str, float]] = []
    for _, _, csv_name, _ in ROWS:
        sub = load_and_filter(SRC / csv_name, HELDOUT_11)
        if len(sub) != len(HELDOUT_11):
            raise RuntimeError(
                f"{csv_name}: macro over {len(sub)}/{len(HELDOUT_11)} countries"
            )
        agg = {c: float(sub[c].mean(skipna=True)) for c in COLS}
        agg["_bnd_n"] = int(sub["boundary_error_m_mean"].notna().sum())
        aggregates.append(agg)

    # Best per column (higher-is-better for PQ/SQ/RQ/AP; lower-is-better
    # for |dN| and boundary errors).
    higher_better = {"pq", "pq_sq", "pq_rq", "ap_5_95"}
    best: dict[str, float] = {}
    for c in COLS:
        vals = [a[c] for a in aggregates]
        best[c] = max(vals) if c in higher_better else min(vals)

    def cell(v: float, c: str, decimals: int = 3) -> str:
        s = f"{v:.{decimals}f}"
        if abs(v - best[c]) < 1e-9:
            s = rf"\textbf{{{s}}}"
        return s

    lines: list[str] = []
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{Panoptic} & & & \multicolumn{2}{c}{Boundary err (m)} \\"
    )
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){8-9}")
    lines.append(
        r"Imagery & Recipe & PQ & SQ & RQ & AP$_{[.5:.95]}$ "
        r"& $|\Delta\,N_{\text{poly}}|$ & mean & p95 \\"
    )
    lines.append(r"\midrule")
    for (imagery, recipe, _, _), agg in zip(ROWS[:4], aggregates[:4]):
        lines.append(
            f"{imagery} & {recipe} & "
            f"{cell(agg['pq'], 'pq')} & {cell(agg['pq_sq'], 'pq_sq')} & "
            f"{cell(agg['pq_rq'], 'pq_rq')} & {cell(agg['ap_5_95'], 'ap_5_95')} & "
            f"{cell(agg['polygon_count_delta_mean'], 'polygon_count_delta_mean', 1)} & "
            f"{cell(agg['boundary_error_m_mean'], 'boundary_error_m_mean', 2)} & "
            f"{cell(agg['boundary_error_m_p95'], 'boundary_error_m_p95', 2)} \\\\"
        )
    lines.append(r"\midrule")
    for (imagery, recipe, _, _), agg in zip(ROWS[4:], aggregates[4:]):
        lines.append(
            f"{imagery} & {recipe} & "
            f"{cell(agg['pq'], 'pq')} & {cell(agg['pq_sq'], 'pq_sq')} & "
            f"{cell(agg['pq_rq'], 'pq_rq')} & {cell(agg['ap_5_95'], 'ap_5_95')} & "
            f"{cell(agg['polygon_count_delta_mean'], 'polygon_count_delta_mean', 1)} & "
            f"{cell(agg['boundary_error_m_mean'], 'boundary_error_m_mean', 2)} & "
            f"{cell(agg['boundary_error_m_p95'], 'boundary_error_m_p95', 2)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for (imagery, recipe, _, _), agg in zip(ROWS, aggregates):
        print(
            f"  {imagery} {recipe}: PQ={agg['pq']:.3f} "
            f"bnd_mean(n={agg['_bnd_n']}/11)={agg['boundary_error_m_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
