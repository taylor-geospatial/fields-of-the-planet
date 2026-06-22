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
import tg_style
from adjustText import (
    adjust_text,  # adjustText not in main CI deps; paper-scripts only
)

# Seaborn notebook context with whitegrid -- subtle grid, larger axis text.
sns.set_theme(context="notebook", style="whitegrid", font="Nimbus Roman", font_scale=0.65)
mpl.rcParams.update(
    {
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.color": "#d9d6c8",
        "axes.edgecolor": tg_style.BROWN,
        "axes.labelcolor": tg_style.BROWN,
        "text.color": tg_style.BROWN,
        "xtick.color": tg_style.BROWN,
        "ytick.color": tg_style.BROWN,
    }
)


OLIVE = tg_style.GREEN_INK
SIENNA = tg_style.RED
NEUTRAL = "#a89f93"
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
    p.add_argument("--out", default="paper/figs/smallholder_scatter.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.src).rename(columns={"median_field_size_ha": "ha"})
    # Recompute delta PQ from the canonical all-23-region B3-full augmax run
    # (planet_b3_augmax_full_22.csv -- the only run covering all 23 regions,
    # giving the 37.9 macro quoted in the per-region prose and matching
    # per_country_bars.py). Earlier this read the stale pq columns baked into
    # the CSV, which came from our original (un-released) B3 checkpoint.
    pq_pl = pd.read_csv("logs/polygon_metrics/planet_b3_augmax_full_22.csv")[
        ["country", "pq"]
    ].rename(columns={"pq": "pq_pl"})
    pq_s2 = pd.read_csv("logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv")[
        ["country", "pq"]
    ].rename(columns={"pq": "pq_s2"})
    df = df.drop(columns=["pq_pl", "pq_s2", "delta_pq"], errors="ignore")
    df = df.merge(pq_pl, on="country", how="inner").merge(pq_s2, on="country", how="inner")
    df["delta_pq"] = df["pq_pl"] - df["pq_s2"]
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
        line_kws={"color": tg_style.BROWN, "linewidth": 0.8},
        ci=95,
    )

    # Scatter points
    ax.scatter(df.log_ha, df.d_f1, c=df.color, s=18, linewidths=0.4, edgecolors="white", zorder=3)
    ax.axhline(0, color=tg_style.BROWN, linewidth=0.4)

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
        texts.append(ax.text(row.log_ha, row.d_f1, row.country_lbl, fontsize=5.8, color="#8a8276"))
    adjust_text(
        texts,
        ax=ax,
        expand=(1.1, 1.2),
        arrowprops={"arrowstyle": "-", "color": "#c4bfb2", "lw": 0.3, "shrinkA": 2, "shrinkB": 2},
    )

    # Log x-axis with hand-formatted ticks (we drew log10-transformed data).
    tick_vals = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks([np.log10(v) for v in tick_vals])
    ax.set_xticklabels([f"{v:g}" for v in tick_vals])
    ax.set_xlim(np.log10(0.08), np.log10(60))
    ax.set_xlabel("Median field area (ha, log scale)")
    ax.set_ylabel(r"$\Delta$ PQ (pp, FTP-PRUE+ $-$ FTW-PRUE+)")
    ax.text(
        0.97,
        0.03,
        f"Pearson $r$ = {r:+.2f}    $n$ = {len(df)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=tg_style.BROWN,
        style="italic",
    )
    ax.tick_params(length=2.5, width=0.4)
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
