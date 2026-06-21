"""Per-region combined panel: Delta PQ (polygon) and Delta Obj F1 (pixel-instance).

Two aligned horizontal-bar panels sharing the same region order (sorted by
Delta PQ). The point is that the two metrics disagree: polygon PQ shows a
broad, consistent PlanetScope advantage, while the FTW-official pixel-instance
object F1 is far noisier and barely favors Planet -- i.e. the pixel-instance
metric understates the resolution benefit, which is why we lead with
polygon-level metrics.

Left panel  -- Delta PQ (FTP-PRUE - FTW-PRUE), polygon recognition quality:
    logs/repro_eval/polygon_metrics_22.csv (released B3-full checkpoint)
    vs logs/polygon_metrics/s2_b7_augmax_full_22.csv
Right panel -- Delta Obj F1 (FTP-PRUE - FTW-PRUE), FTW-official pixel-instance:
    logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv (object_ws_f1)
    vs logs/ftw_official/b7_*.csv (object_level_f1)

Styling copied from make_per_country_bars_pq.py (tg_style palette/fonts).
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


def _load_pq() -> pd.DataFrame:
    pl = pd.read_csv("logs/repro_eval/polygon_metrics_22.csv")[["country", "pq"]].rename(
        columns={"pq": "pq_pl"}
    )
    s2 = pd.read_csv("logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv")[
        ["country", "pq"]
    ].rename(columns={"pq": "pq_s2"})
    m = pl.merge(s2, on="country", how="inner").copy()
    m["d_pq"] = (m.pq_pl - m.pq_s2) * 100.0
    return m[["country", "d_pq"]]


def _load_objf1() -> pd.DataFrame:
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
    return m[["country", "d_f1"]]


def _bars(ax: plt.Axes, y: np.ndarray, vals: np.ndarray, xlabel: str) -> None:
    pos = tg_style.GREEN_INK
    neg = tg_style.RED
    colors = [pos if v >= 0 else neg for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.72)
    ax.axvline(0, color=tg_style.BROWN, linewidth=0.55)
    span = max(vals) - min(vals)
    pad = span * 0.015
    for yi, v in zip(y, vals):
        ax.text(
            v + (pad if v >= 0 else -pad),
            yi,
            f"{v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=6.0,
            color=tg_style.BROWN,
        )
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_xlim(min(vals) - span * 0.18, max(vals) + span * 0.18)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.3, color="#d9d6c8", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figs/per_country_both.pdf")
    args = p.parse_args()

    pq = _load_pq()
    f1 = _load_objf1()
    # Shared region order: sort by Delta PQ. Outer-join so a region missing in
    # one metric (e.g. brazil lacks a full-data PlanetScope Obj F1) still shows
    # in the panel where it exists, with a blank bar in the other.
    m = pq.merge(f1, on="country", how="outer")
    m["sort_key"] = m["d_pq"].fillna(m["d_pq"].min() - 1)
    m = m.sort_values("sort_key", ascending=False).reset_index(drop=True)
    m["country_lbl"] = m.country.str.replace("_", " ").str.title()
    y = np.arange(len(m))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(5.6, 4.2), sharey=True, gridspec_kw={"wspace": 0.06}
    )
    _bars(axL, y, m.d_pq.fillna(0.0).to_numpy(), r"$\Delta$ PQ (pp, polygon)")
    _bars(axR, y, m.d_f1.fillna(0.0).to_numpy(), r"$\Delta$ Obj F1 (pp, pixel-instance)")

    axL.set_yticks(y)
    axL.set_yticklabels(m.country_lbl, fontsize=7.5)
    axL.set_title("Polygon PQ", fontsize=8.5, color=tg_style.BROWN, pad=6)
    axR.set_title("FTW pixel-instance Obj F1", fontsize=8.5, color=tg_style.BROWN, pad=6)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    n_pq = int((m.d_pq > 0).sum())
    n_pq_tot = int(m.d_pq.notna().sum())
    n_f1 = int((m.d_f1 > 0).sum())
    n_f1_tot = int(m.d_f1.notna().sum())
    print(f"PQ wins {n_pq}/{n_pq_tot}; ObjF1 wins {n_f1}/{n_f1_tot}")


if __name__ == "__main__":
    main()
