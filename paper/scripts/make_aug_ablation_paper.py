"""Cumulative augmentation ablation (paper-style, seaborn notebook).

Bars: cumulative Obj F1 (pp) of the PRUE-FTP recipe as we add aug stages.
Dotted reference lines: released PRUE-B3/B7 (S2) numbers.

Minimal styling -- no figure title, single accent color, value labels in
grey, restrained grid.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # seaborn not in main CI deps; paper-scripts only

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="paper/scripts/output/aug_ablation_heldout11.csv")
    p.add_argument("--out", default="paper/figs/aug_ablation_paper.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.src)
    df["label"] = df["label"].str.replace("\n", " ")
    df["pp"] = df["object_ws_f1"] * 100.0

    pl = df[df["panel"] == "planet"].reset_index(drop=True)
    s2 = df[df["panel"] == "s2"].reset_index(drop=True)

    # Palette: progressive olive shades for the cumulative recipe steps.
    base = "#3d5a26"
    n = len(pl)
    fills = [
        # alpha-blend toward white for the early steps; deepest at the right.
        tuple(
            0.55
            + 0.45 * (i / max(1, n - 1)) * np.array([1, 1, 1]) * 0
            + np.array(mpl.colors.to_rgb(base)) * (0.55 + 0.45 * (i / max(1, n - 1)))
        )
        for i in range(n)
    ]
    # Simpler: convert hex to rgb, then scale brightness
    rgb = np.array(mpl.colors.to_rgb(base))
    fills = [
        tuple((rgb + (1 - rgb) * (1 - 0.4 - 0.6 * i / max(1, n - 1))).clip(0, 1)) for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    x = np.arange(n)
    ax.bar(x, pl.pp, color=fills, edgecolor="none", width=0.72, zorder=3)

    # Value labels above each bar
    for xi, v in zip(x, pl.pp):
        ax.text(
            xi,
            v + 0.6,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
            zorder=4,
        )

    # S2 reference lines (dotted)
    ref_styles = {
        "+ augmax, B3 full": ("#c0796b", "PRUE-B3 (S2, full)"),
        "+ augmax, B7 full": ("#7a2e22", "PRUE-B7 (S2, full)"),
        "+ augmax, B3 CC-BY": ("#d6a8a0", "PRUE-B3 (S2, CC-BY)"),
        "+ augmax, B7 CC-BY": ("#9a4e42", "PRUE-B7 (S2, CC-BY)"),
    }
    for _, row in s2.iterrows():
        if row.label in ref_styles:
            color, label = ref_styles[row.label]
            ax.axhline(row.pp, color=color, linestyle=":", linewidth=0.7, zorder=2, label=label)

    leg = ax.legend(
        loc="lower right",
        fontsize=5.8,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=2.0,
        borderpad=0.4,
        labelspacing=0.25,
    )
    leg.get_frame().set_linewidth(0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(pl.label, rotation=22, ha="right", fontsize=6.5)
    ax.set_ylabel("Obj F1 (pp)")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, max(pl.pp.max(), s2.pp.max()) + 7)
    ax.tick_params(length=2.5, width=0.4)
    ax.grid(axis="x", visible=False)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
