"""Per-country bars (paper-style).

Minimal matplotlib version: no title/subtitle, no background tint, no
quadrant shading, no "everything-is-an-accent" coloring.  Just the bars,
a zero reference line, a thin axis, value annotations in light grey, and
country names on the left.  Negative bars in muted red, positive bars in
muted dark green.  Times Roman (Nimbus Roman) to match the rest of the
paper.

Single-column width (3 in) so it drops in alongside body text.
"""

import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tg_style

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.0,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "text.color": tg_style.BROWN,
        "axes.labelcolor": tg_style.BROWN,
        "axes.edgecolor": tg_style.BROWN,
        "xtick.color": tg_style.BROWN,
        "ytick.color": tg_style.BROWN,
        "savefig.bbox": "tight",
    }
)


def _load_deltas() -> pd.DataFrame:
    s2_files = [
        f for f in sorted(glob.glob("logs/ftw_official/b7_*.csv")) if "per_country" not in f
    ]
    s2 = pd.concat([pd.read_csv(f) for f in s2_files], ignore_index=True)
    s2 = (
        s2.rename(columns={"countries": "country", "object_level_f1": "obj_f1"})
        .drop_duplicates(subset="country", keep="last")[["country", "obj_f1"]]
        .rename(columns={"obj_f1": "f1_s2"})
    )
    pl = pd.read_csv("logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv")[
        ["country", "object_ws_f1"]
    ].rename(columns={"object_ws_f1": "f1_pl"})
    m = s2.merge(pl, on="country", how="inner").copy()
    m["d_f1"] = (m.f1_pl - m.f1_s2) * 100.0
    m["country_lbl"] = m.country.str.replace("_", " ").str.title()
    return m.sort_values("d_f1").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figs/per_country_bars_paper.pdf")
    args = p.parse_args()

    df = _load_deltas()
    pos = tg_style.GREEN_INK
    neg = tg_style.RED
    colors = [pos if v >= 0 else neg for v in df.d_f1]

    fig, ax = plt.subplots(figsize=(2.75, 3.3))
    y = np.arange(len(df))
    ax.barh(y, df.d_f1, color=colors, edgecolor="none", height=0.72)

    # Zero reference line
    ax.axvline(0, color=tg_style.BROWN, linewidth=0.55)

    # Numeric annotations at bar tips
    pad = 0.6
    for yi, v in zip(y, df.d_f1):
        ax.text(
            v + (pad if v >= 0 else -pad),
            yi,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=6.5,
            color=tg_style.BROWN,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df.country_lbl, fontsize=7.5)
    ax.set_xlabel(r"$\Delta$ Obj F1 (pp, PRUE-FTP-B3 $-$ PRUE-B7)", labelpad=4)
    ax.set_xlim(min(df.d_f1) - 2.5, max(df.d_f1) + 2.5)
    ax.invert_yaxis()  # largest gain on top
    ax.grid(axis="x", linewidth=0.3, color="#d9d6c8", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=2)
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
