"""Generate ``paper/figs/area_bins.tex`` (``tab:area_bins``).

Polygon metrics broken down by GT field-area bin (small <0.5 ha / medium
0.5--2 ha / large >2 ha) and IoU threshold (RQ at 0.5, PQ, AP[.5:.95]),
pooled (micro) over the 10 dense held-out countries. Each model is scored
against its own-sensor GT (Planet 3 m, S2 10 m); the small bin carries a
footnote because the 10 m rasterization merges ~7k small fields.

Sources: ``logs/area_bins/<stem>.csv.bins.csv`` (from polygon_metrics_eval.py
--area-bins). Run::

    uv run python paper/scripts/make_area_bins_table.py
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figs" / "area_bins.tex"
SRC = REPO / "logs" / "area_bins"

# (display name, backbone, bins-csv stem, bold-row)
ROWS = [
    ("FTP-PRUE+", "B3", "planet_b3", True),
    ("FTW-PRUE+", "B7", "s2_b7", False),
    ("FTW-PRUE+", "B3", "s2_b3", False),
]
BINS = ["all", "small", "medium", "large"]
BIN_HDR = {"all": "All", "small": r"Small$^{\dagger}$", "medium": "Medium", "large": "Large"}
METRICS = [("rq_50", r"RQ$_{.5}$"), ("pq", "PQ"), ("ap_5_95", r"AP")]


def _load(stem: str) -> dict:
    d = pd.read_csv(SRC / f"{stem}.csv.bins.csv").set_index("bin")
    return {b: d.loc[b] for b in BINS}


def main() -> None:
    data = {stem: _load(stem) for _, _, stem, _ in ROWS}
    # best (max) per (bin, metric) across models, for bolding.
    best = {}
    for b in BINS:
        for mk, _ in METRICS:
            best[(b, mk)] = max(data[s][b][mk] for _, _, s, _ in ROWS)

    lines = []
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}ll" + "ccc" * len(BINS) + r"@{}}")
    lines.append(r"\toprule")
    # bin group headers
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{BIN_HDR[b]}}}" for b in BINS)
    lines.append(r"Model & Bb. & " + groups + r" \\")
    cmids = " ".join(rf"\cmidrule(lr){{{3 + i * 3}-{5 + i * 3}}}" for i in range(len(BINS)))
    lines.append(cmids)
    sub = " & ".join(lbl for b in BINS for _, lbl in METRICS)
    lines.append(r" & & " + sub + r" \\")
    lines.append(r"\midrule")
    for model, bb, stem, bold in ROWS:
        cells = []
        for b in BINS:
            for mk, _ in METRICS:
                v = data[stem][b][mk] * 100
                s = f"{v:.1f}"
                if abs(data[stem][b][mk] - best[(b, mk)]) < 1e-9:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
        name = rf"\textbf{{{model}}} & \textbf{{{bb}}}" if bold else f"{model} & {bb}"
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for _, _, stem, _ in ROWS:
        sm = data[stem]["small"]
        print(f"  {stem}: small RQ@.5={sm['rq_50'] * 100:.1f} PQ={sm['pq'] * 100:.1f} n_gt={int(sm['n_gt'])}")


if __name__ == "__main__":
    main()
