"""Generate ``paper/figs/resolution_ablation.tex`` (``tab:resolution_ablation``).

PQ by GT field-size bin for three conditions, scored against the TRUE FTW
polygons under each row's output-grid protocol and macro-averaged over the
10 dense held-out countries (HELDOUT_10_DENSE):

* native 10 m Sentinel-2 (``s2nat10``),
* the same model bilinearly upsampled to a 512-pixel grid (``s2up``),
* real 3 m PlanetScope (``planet3m``).

Reads the per-country ``logs/resolution_ablation/<condition>/<country>.csv.bins.csv``
files at full precision and macro-averages the ``pq`` column per bin, the same
aggregation ``polygon_metrics_table.py`` uses for the PQ-by-size columns of
``tab:polygon_metrics`` -- so the shared rows (s2nat10, planet3m) match Table 1
exactly. (The pre-rounded ``logs/resolution_ablation_macro.csv`` is NOT used: it
stores 2-decimal values, e.g. 15.75, which round to 15.8 rather than the
full-precision 15.7.)

Run::

    uv run python paper/scripts/resolution_ablation_table.py
"""

from pathlib import Path

import pandas as pd
from _aggregate import HELDOUT_10_DENSE

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "resolution_ablation.tex"
RESABL = REPO / "logs" / "resolution_ablation"

AREA_BINS = ("small", "medium", "large")

# (display label, condition dir, bold-row).
ROWS = [
    (r"Sentinel-2 ($10$m, native)", "s2nat10", False),
    (r"Sentinel-2 ($512$, upsampled)", "s2up", False),
    (r"\textbf{PlanetScope ($3$m, real)}", "planet3m", True),
]


def _bins_csvs(condition: str) -> list[Path]:
    d = RESABL / condition
    files = sorted(d.glob("*.csv.bins.csv"))
    if len(files) != len(HELDOUT_10_DENSE):
        raise RuntimeError(
            f"{d}: {len(files)} per-country bins CSVs, expected {len(HELDOUT_10_DENSE)}"
        )
    return files


def _area_pq(condition: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for b in AREA_BINS:
        vals = [float(pd.read_csv(c).set_index("bin").loc[b, "pq"]) for c in _bins_csvs(condition)]
        out[b] = sum(vals) / len(vals)
    return out


def main() -> None:
    pqs = [(_label, bold, _area_pq(cond)) for _label, cond, bold in ROWS]

    def cells(pq: dict[str, float], bold: bool) -> str:
        out = []
        for b in AREA_BINS:
            s = f"{pq[b] * 100:.1f}"
            out.append(rf"\textbf{{{s}}}" if bold else s)
        return " & ".join(out)

    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{PQ by GT field size} \\",
        r"\cmidrule(lr){2-4}",
        r"Condition & \makecell{small\\$<0.5$} & \makecell{med.\\$0.5$--$2$} "
        r"& \makecell{large\\$>2$} \\",
        r"\midrule",
    ]
    for label, bold, pq in pqs:
        lines.append(f"{label} & {cells(pq, bold)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for label, _bold, pq in pqs:
        plain = label.replace(r"\textbf{", "").replace("}", "")
        print(
            f"  {plain:34s} PQ[s/m/l]={pq['small'] * 100:.1f}/"
            f"{pq['medium'] * 100:.1f}/{pq['large'] * 100:.1f}"
        )


if __name__ == "__main__":
    main()
