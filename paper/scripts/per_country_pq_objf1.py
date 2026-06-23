"""Per-region split panels: Delta PQ (polygon) and Delta Obj F1 (pixel-instance).

Two aligned horizontal-bar panels share the same region order (sorted by
Delta PQ). The point is that the two metrics disagree: polygon PQ shows a
broad, consistent PlanetScope advantage, while the FTW-official pixel-instance
object F1 is far noisier and barely favors Planet -- i.e. the pixel-instance
metric understates the resolution benefit, which is why we lead with
polygon-level metrics.

Left panel  -- Delta PQ (FTP-PRUE - FTW-PRUE), polygon recognition quality:
    logs/polygon_metrics/planet_b3_augmax_full_22.csv (all-23-region B3-full augmax run)
    vs logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv
Right panel -- Delta Obj F1 (FTP-PRUE - FTW-PRUE), FTW-official pixel-instance:
    logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv (object_ws_f1; Brazil missing)
    vs logs/ftw_official/b7_*.csv (object_level_f1)

Styling copied from per_country_bars.py (tg_style palette/fonts).
"""

import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tg_style

FALLBACK_DELTAS = [
    ("cambodia", -7.3, -8.6),
    ("germany", -6.8, -7.1),
    ("portugal", -1.6, -2.2),
    ("slovenia", -0.3, -0.0),
    ("vietnam", -0.3, 1.5),
    ("estonia", 0.5, -3.3),
    ("spain", 0.9, -1.1),
    ("croatia", 1.6, 2.0),
    ("france", 3.1, -3.5),
    ("austria", 4.4, 0.9),
    ("south_africa", 4.6, 6.2),
    ("luxembourg", 5.0, -3.2),
    ("corsica", 5.1, -1.0),
    ("finland", 5.3, 3.4),
    ("sweden", 6.7, 4.5),
    ("belgium", 7.3, -1.7),
    ("rwanda", 10.6, 14.3),
    ("latvia", 11.2, 2.2),
    ("slovakia", 11.4, 2.9),
    ("denmark", 11.6, 5.1),
    ("lithuania", 16.3, 5.2),
    ("brazil", 18.3, np.nan),
    ("netherlands", 22.0, -2.8),
]

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
    pl = pd.read_csv("logs/polygon_metrics/planet_b3_augmax_full_22.csv")[["country", "pq"]].rename(
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


def _merged_metrics() -> pd.DataFrame:
    try:
        pq = _load_pq()
        f1 = _load_objf1()
    except (FileNotFoundError, ValueError) as exc:
        print(f"source logs unavailable ({exc}); using embedded plotted deltas")
        m = pd.DataFrame(FALLBACK_DELTAS, columns=["country", "d_pq", "d_f1"])
        m["country_lbl"] = m.country.str.replace("_", " ").str.title()
        return m
    # Shared region order: sort by Delta PQ. Outer-join so a region missing in
    # one metric (e.g. Brazil lacks a full-data PlanetScope Obj F1) still shows
    # in the panel where it exists, with "n/a" in the missing panel.
    m = pq.merge(f1, on="country", how="outer")
    m["sort_key"] = m["d_pq"].fillna(m["d_pq"].min() - 1)
    m = m.sort_values("sort_key", ascending=False).reset_index(drop=True)
    m["country_lbl"] = m.country.str.replace("_", " ").str.title()
    return m


def _bars(ax: plt.Axes, y: np.ndarray, vals: np.ndarray, xlabel: str) -> None:
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    pos = tg_style.GREEN_INK
    neg = tg_style.RED
    colors = [pos if v >= 0 else neg for v in vals[valid]]
    ax.barh(y[valid], vals[valid], color=colors, edgecolor="none", height=0.72)
    ax.axvline(0, color=tg_style.BROWN, linewidth=0.55)
    span = float(np.nanmax(vals) - np.nanmin(vals))
    if span == 0:
        span = 1.0
    pad = span * 0.015
    for yi, v in zip(y[valid], vals[valid]):
        display_v = 0.0 if abs(v) < 0.05 else v
        ax.text(
            v + (pad if v >= 0 else -pad),
            yi,
            f"{display_v:+.1f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=6.0,
            color=tg_style.BROWN,
        )
    for yi in y[~valid]:
        ax.text(
            0,
            yi,
            "n/a",
            va="center",
            ha="center",
            fontsize=6.0,
            color=tg_style.BROWN,
            alpha=0.55,
        )
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_xlim(float(np.nanmin(vals) - span * 0.18), float(np.nanmax(vals) + span * 0.18))
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.3, color="#d9d6c8", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, pad=2)


def _save_split(
    m: pd.DataFrame,
    out: str,
    metric_col: str,
    title: str,
    xlabel: str,
    *,
    show_regions: bool,
) -> None:
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(2.8, 4.2))
    _bars(ax, y, m[metric_col].to_numpy(), xlabel)
    ax.set_yticks(y)
    if show_regions:
        ax.set_yticklabels(m.country_lbl, fontsize=7.5)
    else:
        # Keep invisible labels so the split PDFs have matching tight bboxes.
        ax.set_yticklabels(m.country_lbl, fontsize=7.5, color="white")
    ax.set_title(title, fontsize=8.5, color=tg_style.BROWN, pad=6)
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {out}")


def _save_combined(m: pd.DataFrame, out: str) -> None:
    y = np.arange(len(m))
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(5.6, 4.2), sharey=True, gridspec_kw={"wspace": 0.06}
    )
    _bars(axL, y, m.d_pq.to_numpy(), r"$\Delta$ PQ (pp, polygon)")
    _bars(axR, y, m.d_f1.to_numpy(), r"$\Delta$ Obj F1 (pp, pixel-instance)")

    axL.set_yticks(y)
    axL.set_yticklabels(m.country_lbl, fontsize=7.5)
    axL.set_title("Polygon PQ", fontsize=8.5, color=tg_style.BROWN, pad=6)
    axR.set_title("FTW pixel-instance Obj F1", fontsize=8.5, color=tg_style.BROWN, pad=6)
    axR.tick_params(axis="y", labelleft=False)

    Path(out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figs/per_country_pq_objf1.pdf")
    p.add_argument("--out-pq", default="paper/figs/per_country_pq_split.pdf")
    p.add_argument("--out-objf1", default="paper/figs/per_country_objf1_split.pdf")
    args = p.parse_args()

    m = _merged_metrics()
    _save_combined(m, args.out)
    _save_split(
        m,
        args.out_pq,
        "d_pq",
        "Polygon PQ",
        r"$\Delta$ PQ (pp, polygon)",
        show_regions=True,
    )
    _save_split(
        m,
        args.out_objf1,
        "d_f1",
        "FTW pixel-instance Obj F1",
        r"$\Delta$ Obj F1 (pp, pixel-instance)",
        show_regions=False,
    )
    n_pq = int((m.d_pq > 0).sum())
    n_pq_tot = int(m.d_pq.notna().sum())
    n_f1 = int((m.d_f1 > 0).sum())
    n_f1_tot = int(m.d_f1.notna().sum())
    print(f"PQ wins {n_pq}/{n_pq_tot}; ObjF1 wins {n_f1}/{n_f1_tot}")


if __name__ == "__main__":
    main()
