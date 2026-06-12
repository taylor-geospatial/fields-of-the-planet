"""Per-country deltas: Planet B3 augmax full vs FTW S2 PRUE-B7 full.

Both models evaluated under FTW v3.1 full_data protocol. Top panel: Δ
Obj F1; bottom: Δ Pixel IoU. Bars sorted by Δ Obj F1.

Writes paper/figs/per_country_bars.pdf.
"""

import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

FIGS = Path(__file__).parent.parent / "figs"
FIGS.mkdir(exist_ok=True, parents=True)

s2_files = [f for f in sorted(glob.glob("logs/ftw_official/b7_*.csv")) if "per_country" not in f]
s2 = pd.concat([pd.read_csv(f) for f in s2_files], ignore_index=True)
s2 = s2.rename(columns={"countries": "country", "object_level_f1": "obj_f1"})
s2 = s2.drop_duplicates(subset="country", keep="last")[["country", "pixel_level_iou", "obj_f1"]]
s2 = s2.rename(columns={"pixel_level_iou": "iou_s2", "obj_f1": "f1_s2"})

pl = pd.read_csv("logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv")
pl = pl[["country", "pixel_level_iou", "object_ws_f1"]].rename(
    columns={"pixel_level_iou": "iou_pl", "object_ws_f1": "f1_pl"}
)

m = s2.merge(pl, on="country", how="inner").copy()
# Express everything in percentage points (multiply by 100) for readability.
m["d_iou"] = (m.iou_pl - m.iou_s2) * 100.0
m["d_f1"] = (m.f1_pl - m.f1_s2) * 100.0
m = m.sort_values("d_f1", ascending=True).reset_index(drop=True)

# Two side-by-side panels keep the figure short. Shared y-axis = same
# country order across F1 and IoU.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
y = np.arange(len(m))


def _annotate(ax, vals, xlim, fmt="{:+.1f}"):
    """Place a text label at each bar tip, clipping outliers to the axis edge
    and prefixing them with an arrow so the reader knows they go further."""
    lo, hi = xlim
    pad = (hi - lo) * 0.01
    for yi, v in zip(y, vals):
        if v < lo:
            ax.text(
                lo + pad,
                yi,
                f"← {fmt.format(v)}",
                va="center",
                ha="left",
                fontsize=6,
                color="white",
                fontweight="bold",
            )
        elif v > hi:
            ax.text(
                hi - pad,
                yi,
                f"{fmt.format(v)} →",
                va="center",
                ha="right",
                fontsize=6,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                v + (pad if v >= 0 else -pad),
                yi,
                fmt.format(v),
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=6,
            )


# Left: Δ Obj F1 (percentage points)
colors_f1 = ["#8b3a1f" if d < 0 else "#5b7026" for d in m.d_f1]
xlim_f1 = (-10.0, 16.0)
ax1.barh(y, m.d_f1, color=colors_f1, edgecolor="black", linewidth=0.4)
ax1.axvline(0, color="black", linewidth=0.6)
ax1.set_xlabel(r"$\Delta$ Obj F1 (pp)")
ax1.set_yticks(y)
ax1.set_yticklabels(m.country.str.replace("_", " "))
ax1.grid(axis="x", linewidth=0.4, alpha=0.5)
ax1.set_xlim(*xlim_f1)
_annotate(ax1, m.d_f1, xlim_f1)

# Right: Δ Pixel IoU (percentage points). Kenya/Portugal sit well
# beyond the typical range; clip to a readable window and annotate
# out-of-bounds bars (arrow + value text) so the reader knows the magnitude.
colors_iou = ["#8b3a1f" if d < 0 else "#5b7026" for d in m.d_iou]
xlim_iou = (-15.0, 15.0)
# Visually saturate bars that would extend off the panel; the text
# annotation still reports the true delta.
clipped_iou = np.clip(m.d_iou, xlim_iou[0] + 0.05, xlim_iou[1] - 0.05)
ax2.barh(y, clipped_iou, color=colors_iou, edgecolor="black", linewidth=0.4)
ax2.axvline(0, color="black", linewidth=0.6)
ax2.set_xlabel(r"$\Delta$ Pixel IoU (pp)")
ax2.grid(axis="x", linewidth=0.4, alpha=0.5)
ax2.set_xlim(*xlim_iou)
_annotate(ax2, m.d_iou, xlim_iou)

# Identify and call out the saturated cases in a small caption inside the panel.
outliers = [(c, v) for c, v in zip(m.country, m.d_iou) if v < xlim_iou[0] or v > xlim_iou[1]]
if outliers:
    note = "; ".join(f"{c.replace('_', ' ')}: {v:+.0f} pp" for c, v in outliers)
    ax2.text(
        0.99,
        0.02,
        f"clipped: {note}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        style="italic",
        color="#555555",
    )

fig.suptitle(
    r"FTP (3\,m) vs Sentinel-2 PRUE-B7 full (10\,m): per-country deltas (olive = Planet wins, sienna = S2 wins)",
    fontsize=8,
)
fig.tight_layout(pad=0.4)
fig.savefig(FIGS / "per_country_bars.pdf", bbox_inches="tight")
print(f"wrote {FIGS / 'per_country_bars.pdf'}")
