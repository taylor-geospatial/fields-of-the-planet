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

# (imagery, recipe latex, csv path, midrule-before)
ROWS = [
    ("Planet", r"DelineateAnything (zero-shot)$^{*}$", SRC / "delineate_anything_conf0005.csv", False),
    ("S2", r"\emph{augmax} B3 (CC-BY)", SRC / "s2_b3_augmax_ccby.csv", True),
    ("S2", r"\emph{augmax} B3 (full)", SRC / "s2_b3_augmax_full.csv", False),
    ("S2", r"\emph{augmax} B7 (CC-BY)", SRC / "s2_b7_augmax_ccby.csv", False),
    ("S2", r"\emph{augmax} B7 (full)", SRC / "s2_b7_augmax_full.csv", False),
    ("Planet", r"PRUE-FTP-B3 (CC-BY)", SRC / "planet_b3_augmax_ccby.csv", True),
    ("Planet", r"PRUE-FTP-B7 (CC-BY)", SRC / "planet_b7_augmax_ccby.csv", False),
    ("Planet", r"\textbf{PRUE-FTP-B3 (full)}", REPRO / "polygon_metrics.csv", False),
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
    for _, _, csv_path, _ in ROWS:
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

    def row_line(imagery: str, recipe: str, agg: dict[str, float]) -> str:
        return (
            f"{imagery} & {recipe} & "
            f"{cell(agg['pq'], 'pq')} & {cell(agg['pq_sq'], 'pq_sq')} & "
            f"{cell(agg['pq_rq'], 'pq_rq')} & {cell(agg['ap_5_95'], 'ap_5_95')} & "
            f"{cell(agg['polygon_count_delta_mean'], 'polygon_count_delta_mean', 1)} & "
            f"{cell(agg['boundary_error_m_mean'], 'boundary_error_m_mean', 2)} & "
            f"{cell(agg['boundary_error_m_p95'], 'boundary_error_m_p95', 2)} \\\\"
        )

    lines: list[str] = []
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{@{}l l ccc c c cc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{Panoptic} & & & "
        r"\multicolumn{2}{c}{\makecell{Bd.\ err\ (m)}} \\"
    )
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){8-9}")
    lines.append(
        r"Img. & Recipe & PQ & SQ & RQ & F1$_{[.5{:}.95]}$ & "
        r"\makecell{$|\Delta N|$} & mean & p95 \\"
    )
    lines.append(r"\midrule")
    for (imagery, recipe, _, sep), agg in zip(ROWS, aggregates):
        if sep:
            lines.append(r"\midrule")
        lines.append(row_line(imagery, recipe, agg))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for (imagery, recipe, _, _), agg in zip(ROWS, aggregates):
        print(
            f"  {imagery} {recipe}: PQ={agg['pq']:.3f} "
            f"bnd_mean(n={agg['_bnd_n']}/{len(HELDOUT_10_DENSE)})="
            f"{agg['boundary_error_m_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
