"""Per-region bars: Delta PQ on the FTW full_data 23-region test split.

Same visual style as make_per_country_bars_paper.py, but uses panoptic
quality (PQ) as the comparison metric instead of Obj F1 at IoU=0.5.

Rationale: the paper argues in section 5.4 that PQ is the primary
field-boundary metric because it factors recognition * localization and
is grid-invariant; Obj F1 at IoU=0.5 is recall-sensitive and can flip
the visual story under imagery degradation (Cambodia: -8.6 pp ObjF1
but +6.6 pp PQ). Leading with PQ makes the marquee per-country figure
consistent with the metric the paper champions.

Reads logs/repro_eval/polygon_metrics_22.csv (released B3-full checkpoint)
and logs/polygon_metrics/s2_b7_augmax_full_22.csv; both are the 23-region
full_data test split with WS+TTA inference.
"""

import argparse
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
    # Planet B3-full is the released checkpoint (retrained Jun 2026): repro eval.
    pl = pd.read_csv("logs/repro_eval/polygon_metrics_22.csv")[["country", "pq"]].rename(
        columns={"pq": "pq_pl"}
    )
    s2 = pd.read_csv("logs/polygon_metrics/s2_b7_augmax_full_22.csv")[["country", "pq"]].rename(
        columns={"pq": "pq_s2"}
    )
    m = pl.merge(s2, on="country", how="inner").copy()
    m["d_pq"] = (m.pq_pl - m.pq_s2) * 100.0
    m["country_lbl"] = m.country.str.replace("_", " ").str.title()
    return m.sort_values("d_pq").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figs/per_country_bars_pq.pdf")
    args = p.parse_args()

    df = _load_deltas()
    pos = tg_style.GREEN_INK
    neg = tg_style.RED
    colors = [pos if v >= 0 else neg for v in df.d_pq]

    fig, ax = plt.subplots(figsize=(2.75, 4.0))
    y = np.arange(len(df))
    ax.barh(y, df.d_pq, color=colors, edgecolor="none", height=0.72)
    ax.axvline(0, color=tg_style.BROWN, linewidth=0.55)

    pad = (max(df.d_pq) - min(df.d_pq)) * 0.015
    for yi, v in zip(y, df.d_pq):
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
    ax.set_xlabel(r"$\Delta$ PQ (pp, PRUE-FTP-B3 $-$ S2-augmax-B7)", labelpad=4)
    ax.set_xlim(min(df.d_pq) - 3.0, max(df.d_pq) + 3.0)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.3, color="#d9d6c8", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=2)
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(df[["country", "pq_pl", "pq_s2", "d_pq"]].to_string(index=False))


if __name__ == "__main__":
    main()
