"""Smallholder scatter (paper-style, seaborn ``notebook`` context).

Field-size (log-ha) vs. delta-Obj-F1 (pp) per country. Uses seaborn's
``notebook`` context with ``whitegrid`` style for a clean research-paper
look.  Highlighted countries get bold dot + label; others are quiet grey.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # seaborn not in main CI deps; paper-scripts only
from adjustText import (
    adjust_text,  # adjustText not in main CI deps; paper-scripts only
)

# Seaborn notebook context with whitegrid -- subtle grid, larger axis text.
sns.set_theme(context="notebook", style="whitegrid", font="Nimbus Roman", font_scale=0.65)
mpl.rcParams.update(
    {
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.color": "#dddddd",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    }
)


OLIVE = "#3d5a26"
SIENNA = "#883027"
NEUTRAL = "#a5a39a"
FOCUS = {
    "rwanda",
    "lithuania",
    "south_africa",
    "cambodia",
    "germany",
    "france",
    "denmark",
    "sweden",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="paper/scripts/output/smallholder_scatter.csv")
    p.add_argument("--out", default="paper/figs/smallholder_scatter_paper.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.src).rename(columns={"median_field_size_ha": "ha"})
    df["d_f1"] = df["delta_pq"] * 100.0
    df["country_lbl"] = df["country"].str.replace("_", " ").str.title()
    df["focus"] = df["country"].isin(FOCUS)
    df["color"] = np.where(~df["focus"], NEUTRAL, np.where(df["d_f1"] >= 0, OLIVE, SIENNA))
    df["log_ha"] = np.log10(df["ha"])
    r = np.corrcoef(df["log_ha"], df["d_f1"])[0, 1]

    fig, ax = plt.subplots(figsize=(2.85, 2.3))

    # Regression line + 95% CI using sns.regplot on the log-transformed data.
    sns.regplot(
        data=df,
        x="log_ha",
        y="d_f1",
        ax=ax,
        scatter=False,
        line_kws={"color": "#444444", "linewidth": 0.8},
        ci=95,
    )

    # Scatter points
    ax.scatter(df.log_ha, df.d_f1, c=df.color, s=18, linewidths=0.4, edgecolors="white", zorder=3)
    ax.axhline(0, color="black", linewidth=0.4)

    # Labels — only the focus countries; adjustText handles collision.
    texts = []
    for _, row in df[df["focus"]].iterrows():
        texts.append(
            ax.text(
                row.log_ha,
                row.d_f1,
                row.country_lbl,
                fontsize=6.5,
                color=row.color,
                fontweight="bold",
            )
        )
    for _, row in df[~df["focus"]].iterrows():
        texts.append(ax.text(row.log_ha, row.d_f1, row.country_lbl, fontsize=5.8, color="#888888"))
    adjust_text(
        texts,
        ax=ax,
        expand=(1.1, 1.2),
        arrowprops={"arrowstyle": "-", "color": "#bbbbbb", "lw": 0.3, "shrinkA": 2, "shrinkB": 2},
    )

    # Log x-axis with hand-formatted ticks (we drew log10-transformed data).
    tick_vals = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks([np.log10(v) for v in tick_vals])
    ax.set_xticklabels([f"{v:g}" for v in tick_vals])
    ax.set_xlim(np.log10(0.08), np.log10(60))
    ax.set_xlabel("Median field area (ha, log scale)")
    ax.set_ylabel(r"$\Delta$ PQ (pp, PRUE-HD-B3 $-$ S2-B7)")
    ax.text(
        0.97,
        0.03,
        f"Pearson $r$ = {r:+.2f}    $n$ = {len(df)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
        style="italic",
    )
    ax.tick_params(length=2.5, width=0.4)
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
