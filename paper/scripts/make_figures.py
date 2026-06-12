"""Generate paper figures from wandb runs + eval CSVs.

Outputs:
    paper/figs/training_curves.pdf       - val IoU vs epoch for all 4 runs
    paper/figs/per_country.pdf           - per-country pixel IoU bar chart
    paper/figs/results_table.tex         - LaTeX table snippet, included via input{}
    paper/figs/postproc_delta.pdf        - watershed boost per country (top losers)

Run from repo root:
    uv run python paper/make_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

mpl.rcParams.update(
    {
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
    }
)

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True, parents=True)

# Map wandb run id -> (label, color, linestyle)
RUNS = {
    "1io8addz": ("Baseline (crop 320, $w_b{=}0.75$)", "#3b6aa0", "-"),
    "8y0r6s6r": ("Crop 512 + $w_b{=}0.80$ + dilate 1px", "#c87f3c", "-"),
    "02cqwtmx": (r"Curriculum dilation 3$\rightarrow$0 px", "#6a8c4d", "--"),
    "3e0u1bwd": ("Crop 512 + SDF aux head (best)", "#a73b52", "-"),
}

# --- Figure 1: Training curves -----------------------------------------------
api = wandb.Api()
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for rid, (label, color, ls) in RUNS.items():
    r = api.run(f"isaaccorley/ftw-planet/{rid}")
    hist = list(r.scan_history(keys=["epoch", "val/iou/field"]))
    rows = [(x["epoch"], x["val/iou/field"]) for x in hist if "val/iou/field" in x]
    if not rows:
        continue
    rows.sort()
    eps, iou = zip(*rows)
    ax.plot(eps, iou, color=color, linestyle=ls, linewidth=1.4, label=label)
ax.axhline(0.76, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
ax.text(2, 0.762, "S2 PRUE-B3 finished", fontsize=7, color="black", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val IoU (field, Austria)")
ax.set_xlim(0, 100)
ax.set_ylim(0.0, 0.85)
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
fig.tight_layout(pad=0.4)
fig.savefig(FIGS / "training_curves.pdf", bbox_inches="tight")
print(f"wrote {FIGS / 'training_curves.pdf'}")

# --- Figure 2: Per-country pixel IoU vs S2 PRUE -----------------------------
base = pd.read_csv("logs/postproc_baseline.csv")
sdf = pd.read_csv("logs/postproc_sdf_pad512.csv")
m = (
    base[["country", "pixel_level_iou"]]
    .rename(columns={"pixel_level_iou": "base"})
    .merge(
        sdf[["country", "pixel_level_iou"]].rename(columns={"pixel_level_iou": "sdf"}), on="country"
    )
)
m = m[~m.country.isin(["india", "kenya"])]
m = m.sort_values("sdf", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
y = np.arange(len(m))
ax.barh(y - 0.18, m["base"], height=0.35, color="#3b6aa0", label="PRUE recipe (ours)")
ax.barh(y + 0.18, m["sdf"], height=0.35, color="#a73b52", label="+ crop 512 + SDF (ours)")
ax.axvline(0.76, color="black", linestyle="--", linewidth=0.9, label="S2 PRUE-B3 (mean)")
ax.set_yticks(y)
ax.set_yticklabels(m["country"])
ax.set_xlabel("Pixel IoU (field)")
ax.set_xlim(0, 1)
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="x", linewidth=0.4, alpha=0.5)
fig.tight_layout(pad=0.4)
fig.savefig(FIGS / "per_country.pdf", bbox_inches="tight")
print(f"wrote {FIGS / 'per_country.pdf'}")

# --- Table snippet ----------------------------------------------------------
FULL = {
    "austria",
    "belgium",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
}


def _mean_full(csv_path):
    pp = pd.read_csv(csv_path)
    s = pp[pp.country.isin(FULL)]
    cols = ["pixel_level_iou", "pixel_level_precision", "pixel_level_recall"]
    if "object_ws_f1" in pp.columns:
        cols.append("object_ws_f1")
    elif "object_level_f1" in pp.columns:
        cols.append("object_level_f1")
    return {c: float(s[c].mean()) for c in cols}


base_pp = _mean_full("logs/postproc_baseline.csv")  # PRUE-B3 baseline + WS + pad=mult32
sdf_pp = _mean_full("logs/postproc_sdf_pad512.csv")  # SDF model + WS + pad>=512

table = rf"""\begin{{tabular}}{{lcccc}}
\toprule
Model & Pix IoU & Pix Prec & Pix Rec & Obj F1 \\
\midrule
S2 PRUE-B3 \cite{{kerner2024ftw}} & 0.76 & 0.87 & 0.86 & -- \\
S2 PRUE-B5 \cite{{kerner2024ftw}} & 0.76 & 0.88 & 0.86 & -- \\
S2 PRUE-B7 \cite{{kerner2024ftw}} & 0.77 & 0.88 & 0.86 & -- \\
\midrule
FTP B3 (PRUE recipe)            & {base_pp["pixel_level_iou"]:.3f} & {base_pp["pixel_level_precision"]:.3f} & {base_pp["pixel_level_recall"]:.3f} & {base_pp["object_ws_f1"]:.3f} \\
\textbf{{+ crop 512 + SDF aux head (ours)}}  & \textbf{{{sdf_pp["pixel_level_iou"]:.3f}}} & \textbf{{{sdf_pp["pixel_level_precision"]:.3f}}} & \textbf{{{sdf_pp["pixel_level_recall"]:.3f}}} & \textbf{{{sdf_pp["object_ws_f1"]:.3f}}} \\
\bottomrule
\end{{tabular}}
"""
(FIGS / "results_table.tex").write_text(table)
print(f"wrote {FIGS / 'results_table.tex'}")
print(
    f"baseline (PRUE recipe) iou: {base_pp['pixel_level_iou']:.3f}  objF1_ws: {base_pp['object_ws_f1']:.3f}"
)
print(
    f"SDF model (crop512+SDF)    iou: {sdf_pp['pixel_level_iou']:.3f}  objF1_ws: {sdf_pp['object_ws_f1']:.3f}"
)
