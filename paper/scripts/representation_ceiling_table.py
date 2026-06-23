"""Generate ``paper/figs/representation_ceiling.tex`` (``tab:representation_ceiling``).

Model-free recovery ceiling: the fraction of true FTW fields recoverable as
distinct polygons (IoU > 0.5 vs the true polygon) when the true polygons are
rasterized at each native resolution (3 m vs 10 m) and vectorized back, with a
1-pixel boundary between touching parcels. Pooled over the 10 dense held-out
regions; no model or checkpoint involved.

Reads the POOLED rows of ``logs/representation_ceiling.csv`` (written by
``scripts/eval/representation_ceiling.py``).

Run::

    uv run python paper/scripts/representation_ceiling_table.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "representation_ceiling.tex"
SRC = REPO / "logs" / "representation_ceiling.csv"

# (display label, bin key, bold the 3 m cell).
ROWS = [
    (r"Small ($<0.5$\,ha)", "small", True),
    (r"Medium ($0.5$--$2$\,ha)", "medium", False),
    (r"Large ($>2$\,ha)", "large", False),
]


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> None:
    df = pd.read_csv(SRC)
    pooled = df[df["region"] == "POOLED"].set_index("bin")

    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrcc@{}}",
        r"\toprule",
        r"GT field size & $n$ & 3\,m & 10\,m \\",
        r"\midrule",
    ]
    for label, b, bold in ROWS:
        n = int(pooled.loc[b, "n_true"])
        r3 = float(pooled.loc[b, "recall_3m"]) * 100
        r10 = float(pooled.loc[b, "recall_10m"]) * 100
        c3 = rf"\textbf{{{r3:.1f}}}" if bold else f"{r3:.1f}"
        lines.append(f"{label} & {_thousands(n)} & {c3} & {r10:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for label, b, _bold in ROWS:
        plain = label.replace(r"\,", " ")
        print(f"  {plain:24s} n={int(pooled.loc[b, 'n_true']):6d} "
              f"3m={float(pooled.loc[b, 'recall_3m']) * 100:.1f} "
              f"10m={float(pooled.loc[b, 'recall_10m']) * 100:.1f}")


if __name__ == "__main__":
    main()
