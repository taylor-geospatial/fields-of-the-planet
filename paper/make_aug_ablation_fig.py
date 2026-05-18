"""Aug-stack ablation figure.

Bars per recipe stage on the 9-country held-out set (Obj F1 + WS+TTA where applicable):

  PRUE recipe (no extra augs) -> + preproc/resize -> + swap+gamma (augplus)
    -> + bespoke bundle (augmax)

One bar group per imagery source (Planet, S2). Reference horizontal lines for
the FTW v3.1 released S2 PRUE checkpoints (CC-BY B3/B7 and full B7).

Writes paper/figs/aug_ablation.pdf.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True, parents=True)

# Hard-coded from logs/heldout/*.csv + logs/postproc_ablation/*.csv (best obj F1 WS).
# Numbers are macro-avg over the 9 dense-label held-out countries (excl. K+P).
PLANET = {
    "PRUE\n(no augs)":     0.323,
    "+ preproc / resize":  0.355,
    "+ swap + gamma\n(augplus)": 0.365,
    "+ bespoke bundle\n(augmax, B3 CC-BY)": 0.415,
    "+ augmax, B3 full": 0.495,
}
S2 = {
    "+ bespoke bundle\n(augmax, B3 CC-BY)": 0.370,
    "+ augmax, B3 full":  0.388,
    "+ augmax, B7 CC-BY": 0.386,
    "+ augmax, B7 full":  0.435,
}

# FTW v3.1 released reference (S2 PRUE)
REF = {
    "S2 PRUE-B3 (CC-BY) ref": 0.39,
    "S2 PRUE-B7 (CC-BY) ref": 0.44,
    "S2 PRUE-B7 full ref":    0.47,
}

fig, (axP, axS) = plt.subplots(1, 2, figsize=(11.0, 3.4), sharey=True)

# --- Planet panel ---
labels = list(PLANET)
vals = list(PLANET.values())
colors = ["#dccfb0", "#b8c19d", "#9aa17a", "#6b7d3d", "#3d4f1c"]
xs = np.arange(len(labels))
axP.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.4)
for x, v in zip(xs, vals):
    axP.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
axP.set_title("FTW-HD (3\\,m)")
axP.set_ylabel("Obj F1 (WS + TTA, 9-country held-out)")
axP.set_xticks(xs)
axP.set_xticklabels(labels, rotation=20, ha="right")
axP.set_ylim(0.0, 0.60)
axP.grid(axis="y", linewidth=0.4, alpha=0.5)
# Reference lines
axP.axhline(REF["S2 PRUE-B3 (CC-BY) ref"], color="#888", linestyle=":", linewidth=0.8)
axP.text(0.02, REF["S2 PRUE-B3 (CC-BY) ref"] + 0.005, "S2 B3 CC-BY ref (0.39)", fontsize=7, color="#666", transform=axP.get_yaxis_transform())
axP.axhline(REF["S2 PRUE-B7 full ref"], color="#a44", linestyle=":", linewidth=0.8)
axP.text(0.02, REF["S2 PRUE-B7 full ref"] + 0.005, "S2 B7 full ref (0.47)", fontsize=7, color="#a44", transform=axP.get_yaxis_transform())

# --- S2 panel ---
labels = list(S2)
vals = list(S2.values())
colors = ["#6b7d3d", "#3d4f1c", "#a85a2c", "#8b3a1f"]
xs = np.arange(len(labels))
axS.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.4)
for x, v in zip(xs, vals):
    axS.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
axS.set_title("Sentinel-2 (10\\,m)")
axS.set_xticks(xs)
axS.set_xticklabels(labels, rotation=20, ha="right")
axS.grid(axis="y", linewidth=0.4, alpha=0.5)
axS.axhline(REF["S2 PRUE-B3 (CC-BY) ref"], color="#888", linestyle=":", linewidth=0.8)
axS.text(0.02, REF["S2 PRUE-B3 (CC-BY) ref"] + 0.005, "S2 B3 CC-BY ref", fontsize=7, color="#666", transform=axS.get_yaxis_transform())
axS.axhline(REF["S2 PRUE-B7 (CC-BY) ref"], color="#888", linestyle="--", linewidth=0.8)
axS.text(0.02, REF["S2 PRUE-B7 (CC-BY) ref"] + 0.005, "S2 B7 CC-BY ref", fontsize=7, color="#666", transform=axS.get_yaxis_transform())
axS.axhline(REF["S2 PRUE-B7 full ref"], color="#a44", linestyle=":", linewidth=0.8)
axS.text(0.02, REF["S2 PRUE-B7 full ref"] + 0.005, "S2 B7 full ref", fontsize=7, color="#a44", transform=axS.get_yaxis_transform())

fig.tight_layout(pad=0.4)
fig.savefig(FIGS / "aug_ablation.pdf", bbox_inches="tight")
print(f"wrote {FIGS / 'aug_ablation.pdf'}")
