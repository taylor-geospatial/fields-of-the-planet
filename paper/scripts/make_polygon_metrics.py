"""Generate ``paper/figs/polygon_metrics.tex`` (``tab:polygon_metrics``).

Dense-label held-out macro-average (HELDOUT_10_DENSE: the 11 held-out
countries minus kenya) of panoptic-quality, F1, polygon-count delta, and
meter-scale boundary-error metrics for each (imagery, recipe) configuration.
Sources: ``logs/polygon_metrics/<stem>.csv``.

Kenya's labels are presence-only (background untrusted), so its supervised
polygon metrics are not comparable and it is excluded from the macro; see
the per-country table for the presence-only stress-test row. Boundary-error
entries for kenya are NaN in the source CSVs regardless.

Run::

    uv run python paper/scripts/make_polygon_metrics.py
"""

from pathlib import Path

from _aggregate import HELDOUT_10_DENSE, load_and_filter

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "polygon_metrics.tex"
SRC = REPO / "logs" / "polygon_metrics"

# The B3-full row is the released checkpoint (retrained Jun 2026, epoch 92);
# its metrics come from the reproduction eval rather than the original run.
REPRO = REPO / "logs" / "repro_eval"

# (model, backbone, split, csv path, midrule-before, bold-row)
ROWS = [
    ("DelineateAnything (zero-shot)$^{*}$", "--", "--", SRC / "delineate_anything_conf0005.csv", False, False),
    ("FTW-PRUE", "B3", "CC-BY", SRC / "s2_b3_augmax_ccby.csv", True, False),
    ("FTW-PRUE", "B3", "full", SRC / "s2_b3_augmax_full.csv", False, False),
    ("FTW-PRUE", "B7", "CC-BY", SRC / "s2_b7_augmax_ccby.csv", False, False),
    ("FTW-PRUE", "B7", "full", SRC / "s2_b7_augmax_full.csv", False, False),
    ("FTP-PRUE", "B3", "CC-BY", SRC / "planet_b3_augmax_ccby.csv", True, False),
    ("FTP-PRUE", "B7", "CC-BY", SRC / "planet_b7_augmax_ccby.csv", False, False),
    ("FTP-PRUE", "B3", "full", REPRO / "polygon_metrics.csv", False, True),
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
    for *_, csv_path, _, _ in ROWS:
        sub = load_and_filter(csv_path, HELDOUT_10_DENSE)
        if len(sub) != len(HELDOUT_10_DENSE):
            raise RuntimeError(
                f"{csv_path}: macro over {len(sub)}/{len(HELDOUT_10_DENSE)} countries"
            )
        agg = {c: float(sub[c].mean(skipna=True)) for c in COLS}
        agg["_bnd_n"] = int(sub["boundary_error_m_mean"].notna().sum())
        aggregates.append(agg)

    # Best per column (higher-is-better for PQ/SQ/RQ/F1; lower-is-better
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

    def row_line(model: str, backbone: str, split: str, bold: bool, agg: dict[str, float]) -> str:
        m, b, s = (rf"\textbf{{{x}}}" for x in (model, backbone, split)) if bold else (model, backbone, split)
        return (
            f"{m} & {b} & {s} & "
            f"{cell(agg['pq'], 'pq')} & {cell(agg['pq_sq'], 'pq_sq')} & "
            f"{cell(agg['pq_rq'], 'pq_rq')} & {cell(agg['ap_5_95'], 'ap_5_95')} & "
            f"{cell(agg['polygon_count_delta_mean'], 'polygon_count_delta_mean', 1)} & "
            f"{cell(agg['boundary_error_m_mean'], 'boundary_error_m_mean', 2)} & "
            f"{cell(agg['boundary_error_m_p95'], 'boundary_error_m_p95', 2)} \\\\"
        )

    lines: list[str] = []
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{@{}l l l ccc c c cc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& & & \multicolumn{3}{c}{Panoptic} & & & "
        r"\multicolumn{2}{c}{\makecell{Bd.\ err\ (m)}} \\"
    )
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){9-10}")
    lines.append(
        r"Model & Backbone & Split & PQ & SQ & \makecell{RQ\\($=$F1$_{.5}$)} & "
        r"F1$_{[.5{:}.95]}$ & \makecell{$|\Delta N|$} & mean & p95 \\"
    )
    lines.append(r"\midrule")
    for (model, backbone, split, _, sep, bold), agg in zip(ROWS, aggregates):
        if sep:
            lines.append(r"\midrule")
        lines.append(row_line(model, backbone, split, bold, agg))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for (model, backbone, _, _, _, _), agg in zip(ROWS, aggregates):
        print(
            f"  {model} {backbone}: PQ={agg['pq']:.3f} "
            f"bnd_mean(n={agg['_bnd_n']}/{len(HELDOUT_10_DENSE)})="
            f"{agg['boundary_error_m_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
