"""Per-region split panels: Delta PQ, Delta Obj F1, and Delta small-field PQ.

Three aligned horizontal-bar panels share the same region order (sorted by
Delta small-field PQ). The point is that the metrics disagree by margin and breadth:
polygon PQ shows a broad, consistent PlanetScope advantage; the FTW-official
pixel-instance object F1 is far noisier and barely favors Planet (it understates
the resolution benefit); and small-field (<0.5 ha) polygon PQ shows the largest,
most uniform advantage -- PlanetScope wins all but one region (Portugal ties) --
which is the smallholder-resolution story sharpened to the per-region level.

Panels 1 and 3 score both sensors against the TRUE FTW vector polygons at each
sensor's native ground resolution (Planet 3 m; Sentinel-2 capped to 10 m), the
same honest-footing protocol as tab:resolution_ablation -- so S2 is not credited
for sub-10 m boundaries it cannot resolve.

Panel 1 -- Delta PQ (FTP-PRUE - FTW-PRUE), polygon recognition quality:
    logs/area_bins_per_country/planet_b3_full23_overall_truegt.csv (Planet 3 m)
    vs logs/area_bins_per_country/s2_b7_full23_overall_truegt_gsd10.csv (S2 native 10 m)
Panel 2 -- Delta Obj F1 (FTP-PRUE - FTW-PRUE), FTW-official pixel-instance:
    logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv (object_ws_f1; Brazil missing)
    vs logs/ftw_official/b7_*.csv (object_level_f1)
Panel 3 -- Delta small-field PQ (<0.5 ha GT fields), same true-GT protocol as Panel 1:
    logs/area_bins_per_country/planet_b3_full23_small_truegt.csv
    vs logs/area_bins_per_country/s2_b7_full23_small_truegt_gsd10.csv

Styling via the shared tg_style palette/fonts.
"""

import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tg_style

# (country, d_pq, d_f1, d_pq_small) -- embedded plotted deltas so the figure
# reproduces without the gitignored raw logs. d_pq_small is per-region
# PlanetScope - S2-B7 polygon PQ on <0.5 ha GT fields (points).
FALLBACK_DELTAS = [
    ("finland", 21.6, 3.4, 18.3),
    ("lithuania", 29.1, 5.2, 17.1),
    ("netherlands", 34.2, -2.8, 17.0),
    ("france", 18.4, -3.5, 15.2),
    ("luxembourg", 21.3, -3.2, 15.2),
    ("sweden", 16.6, 4.5, 12.8),
    ("denmark", 17.0, 5.1, 10.7),
    ("austria", 20.8, 0.9, 10.5),
    ("belgium", 16.6, -1.7, 10.4),
    ("slovakia", 13.2, 2.9, 10.3),
    ("latvia", 19.7, 2.2, 10.2),
    ("croatia", 14.0, 2.0, 10.2),
    ("south_africa", 5.0, 6.2, 10.0),
    ("corsica", 10.3, -1.0, 5.6),
    ("estonia", 8.0, -3.3, 5.5),
    ("germany", 2.0, -7.1, 4.6),
    ("slovenia", 8.3, -0.0, 4.2),
    ("spain", 7.1, -1.1, 4.1),
    ("brazil", 30.9, np.nan, 3.4),
    ("cambodia", 1.7, -8.6, 3.1),
    ("rwanda", 11.5, 14.3, 2.1),
    ("vietnam", 3.2, 1.5, 1.7),
    ("portugal", -0.1, -2.2, -0.1),
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
    """Per-region Delta PQ over all fields (PlanetScope 3 m - S2-B7 native 10 m).

    True-GT, native-GSD scoring (same protocol as Panel 3 and
    tab:resolution_ablation): S2 capped to 10 m so it is not credited for
    sub-10 m boundaries it cannot resolve.
    """
    pl = pd.read_csv("logs/area_bins_per_country/planet_b3_full23_overall_truegt.csv")[
        ["country", "pq"]
    ].rename(columns={"pq": "pq_pl"})
    s2 = pd.read_csv("logs/area_bins_per_country/s2_b7_full23_overall_truegt_gsd10.csv")[
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


def _load_pq_small() -> pd.DataFrame:
    """Per-region Delta PQ on <0.5 ha GT fields (PlanetScope 3 m - S2-B7 native 10 m).

    True-GT, native-GSD scoring (same protocol as Panel 1).
    """
    pl = pd.read_csv("logs/area_bins_per_country/planet_b3_full23_small_truegt.csv")[
        ["country", "pq_small"]
    ].rename(columns={"pq_small": "pq_pl"})
    s2 = pd.read_csv("logs/area_bins_per_country/s2_b7_full23_small_truegt_gsd10.csv")[
        ["country", "pq_small"]
    ].rename(columns={"pq_small": "pq_s2"})
    m = pl.merge(s2, on="country", how="inner").copy()
    m["d_pq_small"] = (m.pq_pl - m.pq_s2) * 100.0
    return m[["country", "d_pq_small"]]


def _merged_metrics() -> pd.DataFrame:
    try:
        pq = _load_pq()
        f1 = _load_objf1()
        pqs = _load_pq_small()
        # Outer-join so a region missing in one metric (e.g. Brazil lacks a
        # full-data PlanetScope Obj F1) still shows in the panels where it
        # exists, marked "n/a" elsewhere.
        m = pq.merge(f1, on="country", how="outer").merge(pqs, on="country", how="outer")
    except (FileNotFoundError, ValueError) as exc:
        print(f"source logs unavailable ({exc}); using embedded plotted deltas")
        m = pd.DataFrame(FALLBACK_DELTAS, columns=["country", "d_pq", "d_f1", "d_pq_small"])
    # Shared region order across all panels: sort by Delta small-field PQ.
    m["sort_key"] = m["d_pq_small"].fillna(m["d_pq_small"].min() - 1)
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
    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(7.2, 4.2), sharey=True, gridspec_kw={"wspace": 0.06}
    )
    _bars(axL, y, m.d_pq_small.to_numpy(), r"$\Delta$ PQ (points, $<0.5$ ha fields)")
    _bars(axM, y, m.d_pq.to_numpy(), r"$\Delta$ PQ (points, polygon)")
    _bars(axR, y, m.d_f1.to_numpy(), r"$\Delta$ Obj F1 (points, pixel-instance)")

    axL.set_yticks(y)
    axL.set_yticklabels(m.country_lbl, fontsize=7.5)
    axL.set_title("Small-field polygon PQ", fontsize=8.5, color=tg_style.BROWN, pad=6)
    axM.set_title("Polygon PQ", fontsize=8.5, color=tg_style.BROWN, pad=6)
    axR.set_title("Pixel Obj F1", fontsize=8.5, color=tg_style.BROWN, pad=6)
    axM.tick_params(axis="y", labelleft=False)
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
    p.add_argument("--out-pq-small", default="paper/figs/per_country_pq_small_split.pdf")
    args = p.parse_args()

    m = _merged_metrics()
    _save_combined(m, args.out)
    _save_split(
        m,
        args.out_pq_small,
        "d_pq_small",
        "Small-field polygon PQ",
        r"$\Delta$ PQ (points, $<0.5$ ha fields)",
        show_regions=True,
    )
    _save_split(
        m,
        args.out_pq,
        "d_pq",
        "Polygon PQ",
        r"$\Delta$ PQ (points, polygon)",
        show_regions=False,
    )
    _save_split(
        m,
        args.out_objf1,
        "d_f1",
        "Pixel Obj F1",
        r"$\Delta$ Obj F1 (points, pixel-instance)",
        show_regions=False,
    )
    n_pq = int((m.d_pq > 0).sum())
    n_pq_tot = int(m.d_pq.notna().sum())
    n_f1 = int((m.d_f1 > 0).sum())
    n_f1_tot = int(m.d_f1.notna().sum())
    n_pqs = int((m.d_pq_small > 0).sum())
    n_pqs_tot = int(m.d_pq_small.notna().sum())
    print(
        f"PQ wins {n_pq}/{n_pq_tot}; ObjF1 wins {n_f1}/{n_f1_tot}; small-PQ wins {n_pqs}/{n_pqs_tot}"
    )


if __name__ == "__main__":
    main()
