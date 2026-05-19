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

FIGS = Path(__file__).parent / "figs"
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
m["d_iou"] = m.iou_pl - m.iou_s2
m["d_f1"] = m.f1_pl - m.f1_s2
m = m.sort_values("d_f1", ascending=True).reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
y = np.arange(len(m))

# Top: Δ Obj F1
colors_f1 = ["#8b3a1f" if d < 0 else "#5b7026" for d in m.d_f1]
ax1.barh(y, m.d_f1, color=colors_f1, edgecolor="black", linewidth=0.4)
ax1.axvline(0, color="black", linewidth=0.6)
ax1.set_xlabel(r"$\Delta$ Obj F1 (Planet B3 augmax full $-$ S2 PRUE-B7 full)")
ax1.set_yticks(y)
ax1.set_yticklabels(m.country.str.replace("_", " "))
ax1.grid(axis="x", linewidth=0.4, alpha=0.5)
for yi, v in zip(y, m.d_f1):
    ax1.text(
        v + (0.003 if v >= 0 else -0.003),
        yi,
        f"{v:+.3f}",
        va="center",
        ha="left" if v >= 0 else "right",
        fontsize=7,
    )
ax1.set_xlim(-0.1, 0.16)

# Bottom: Δ Pixel IoU (same country order)
colors_iou = ["#8b3a1f" if d < 0 else "#5b7026" for d in m.d_iou]
ax2.barh(y, m.d_iou, color=colors_iou, edgecolor="black", linewidth=0.4)
ax2.axvline(0, color="black", linewidth=0.6)
ax2.set_xlabel(r"$\Delta$ Pixel IoU (Planet B3 augmax full $-$ S2 PRUE-B7 full)")
ax2.set_yticks(y)
ax2.set_yticklabels(m.country.str.replace("_", " "))
ax2.grid(axis="x", linewidth=0.4, alpha=0.5)
for yi, v in zip(y, m.d_iou):
    ax2.text(
        v + (0.005 if v >= 0 else -0.005),
        yi,
        f"{v:+.3f}",
        va="center",
        ha="left" if v >= 0 else "right",
        fontsize=7,
    )
ax2.set_xlim(-0.55, 0.15)

fig.suptitle(
    "FTW-Planet (3 m) vs Sentinel-2 PRUE-B7 full (10 m) --- full_data per-country deltas\n"
    "(blue = Planet wins, red = Planet loses)",
    fontsize=9,
)
fig.tight_layout(pad=0.4)
fig.savefig(FIGS / "per_country_bars.pdf", bbox_inches="tight")
print(f"wrote {FIGS / 'per_country_bars.pdf'}")
