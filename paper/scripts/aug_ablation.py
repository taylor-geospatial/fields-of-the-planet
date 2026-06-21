"""Cumulative augmentation ablation (paper-style, seaborn notebook).

Bars: cumulative Obj F1 (pp) of the FTP-PRUE recipe as we add aug stages.
Dotted reference lines: released FTW-PRUE (B3/B7) numbers.

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
import tg_style

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="paper/scripts/output/aug_ablation_heldout10.csv")
    p.add_argument("--out", default="paper/figs/aug_ablation.pdf")
    args = p.parse_args()

    df = pd.read_csv(args.src)
    df["label"] = df["label"].str.replace("\n", " ")
    df["pp"] = df["object_ws_f1"] * 100.0

    pl = df[df["panel"] == "planet"].reset_index(drop=True)
    s2 = df[df["panel"] == "s2"].reset_index(drop=True)

    # Cumulative recipe steps ramp from pale to deep brand red (our result).
    n = len(pl)
    rgb = np.array(mpl.colors.to_rgb(tg_style.RED))
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
            color=tg_style.BROWN,
            zorder=4,
        )

    # S2 reference lines: the two same-recipe S2 PRUE+ baselines on the
    # full split (B3 = architecture-matched, B7 = best S2). The CC-BY S2
    # numbers are reported in the caption instead of plotted -- earlier all
    # four lines were drawn and two of them coincided at ~34.8 pp, which is
    # what made the panel unreadable. Distinct color + linestyle + a value
    # label at the right margin keep the two remaining lines unambiguous.
    ref_styles = {
        "B3 full": ("#5a7ab8", "-", "FTW-PRUE (B3)"),
        "B7 full": ("#1f3a6b", "--", "FTW-PRUE (B7)"),
    }
    for _, row in s2.iterrows():
        if row.label in ref_styles:
            color, ls, label = ref_styles[row.label]
            ax.axhline(row.pp, color=color, linestyle=ls, linewidth=1.0, zorder=2, label=label)

    leg = ax.legend(
        loc="lower right",
        fontsize=5.8,
        framealpha=0.9,
        edgecolor="#d9d6c8",
        handlelength=2.4,
        borderpad=0.4,
        labelspacing=0.3,
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
