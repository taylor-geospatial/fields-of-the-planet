"""Aug-stack ablation figure + macro-avg tables.

Bars per recipe stage on the held-out set (Obj F1 + WS+TTA where applicable):

  PRUE recipe (no extra augs) -> + preproc/resize -> + swap+gamma (augplus)
    -> + bespoke bundle (augmax)

One bar group per imagery source (Planet, S2). Reference horizontal lines for
the FTW v3.1 released S2 PRUE checkpoints (CC-BY B3/B7 and full B7).

Country sets are parameterized via ``--countries`` (``heldout11`` or
``heldout9``). The script emits CSVs for *both* the 11-country (headline) and
9-country (diagnostic) macro-averages regardless of ``--countries``; the
``--countries`` flag only selects which numbers drive the figure bars.

Writes:
  paper/scripts/figs/aug_ablation.pdf
  paper/scripts/output/aug_ablation_heldout11.csv
  paper/scripts/output/aug_ablation_heldout9.csv
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _aggregate import HELDOUT_9, HELDOUT_11, aggregate_table

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

HERE = Path(__file__).parent
FIGS = HERE / "figs"
OUT = HERE / "output"
FIGS.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

# Repo root (paper/scripts/.. = repo root)
REPO = HERE.parent.parent

# Bar -> source CSV (object Obj F1 with WS+TTA where the CSV has it).
PLANET_ROWS: list[tuple[str, Path]] = [
    ("PRUE\n(no augs)", REPO / "logs/heldout/b3base_best.csv"),
    ("+ preproc / resize", REPO / "logs/heldout/b3base_aug_best.csv"),
    ("+ swap + gamma\n(augplus)", REPO / "logs/heldout/v3_augplus.csv"),
    ("+ bespoke bundle\n(augmax, B3 CC-BY)", REPO / "logs/heldout/v3_augmax_ws_tta.csv"),
    ("+ augmax, B3 full", REPO / "logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv"),
]
S2_ROWS: list[tuple[str, Path]] = [
    (
        "+ bespoke bundle\n(augmax, B3 CC-BY)",
        REPO / "logs/fulldata_eval/s2_b3_augmax_ccby_ws_tta.csv",
    ),
    ("+ augmax, B3 full", REPO / "logs/fulldata_eval/s2_b3_augmax_full_ws_tta.csv"),
    ("+ augmax, B7 CC-BY", REPO / "logs/fulldata_eval/s2_b7_augmax_ccby_ws_tta.csv"),
    ("+ augmax, B7 full", REPO / "logs/fulldata_eval/s2_b7_augmax_full_ws_tta.csv"),
]

# FTW v3.1 released reference (S2 PRUE), unchanged hand-copied values.
REF = {
    "S2 PRUE-B3 (CC-BY) ref": 0.39,
    "S2 PRUE-B7 (CC-BY) ref": 0.44,
    "S2 PRUE-B7 full ref": 0.47,
}

METRIC = "object_ws_f1"


def _build_tables(
    countries: tuple[str, ...], tag: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute Planet+S2 aggregate tables, persist CSV, return label->Obj-F1."""
    planet_df = aggregate_table([(lbl, p) for lbl, p in PLANET_ROWS], countries, metrics=(METRIC,))
    planet_df.insert(0, "panel", "planet")
    s2_df = aggregate_table([(lbl, p) for lbl, p in S2_ROWS], countries, metrics=(METRIC,))
    s2_df.insert(0, "panel", "s2")
    full = pd.concat([planet_df, s2_df], ignore_index=True)
    csv_path = OUT / f"aug_ablation_{tag}.csv"
    full.to_csv(csv_path, index=False)
    print(f"\n=== {tag} macro-avg ({len(countries)} countries) ===")
    print(full.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {csv_path}")
    planet_vals = dict(zip(planet_df["label"], planet_df[METRIC]))
    s2_vals = dict(zip(s2_df["label"], s2_df[METRIC]))
    return planet_vals, s2_vals


def _draw(planet_vals: dict[str, float], s2_vals: dict[str, float], country_label: str) -> None:
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(11.0, 3.4), sharey=True)

    # Planet panel
    labels = list(planet_vals)
    vals = list(planet_vals.values())
    colors = ["#dccfb0", "#b8c19d", "#9aa17a", "#6b7d3d", "#3d4f1c"]
    xs = np.arange(len(labels))
    axP.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.4)
    for x, v in zip(xs, vals):
        axP.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    axP.set_title("FTW-HD (3\\,m)")
    axP.set_ylabel(f"Obj F1 (WS + TTA, {country_label} held-out)")
    axP.set_xticks(xs)
    axP.set_xticklabels(labels, rotation=20, ha="right")
    axP.set_ylim(0.0, 0.60)
    axP.grid(axis="y", linewidth=0.4, alpha=0.5)
    axP.axhline(REF["S2 PRUE-B3 (CC-BY) ref"], color="#888", linestyle=":", linewidth=0.8)
    axP.text(
        0.02,
        REF["S2 PRUE-B3 (CC-BY) ref"] + 0.005,
        "S2 B3 CC-BY ref (0.39)",
        fontsize=7,
        color="#666",
        transform=axP.get_yaxis_transform(),
    )
    axP.axhline(REF["S2 PRUE-B7 full ref"], color="#a44", linestyle=":", linewidth=0.8)
    axP.text(
        0.02,
        REF["S2 PRUE-B7 full ref"] + 0.005,
        "S2 B7 full ref (0.47)",
        fontsize=7,
        color="#a44",
        transform=axP.get_yaxis_transform(),
    )

    # S2 panel
    labels = list(s2_vals)
    vals = list(s2_vals.values())
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
    axS.text(
        0.02,
        REF["S2 PRUE-B3 (CC-BY) ref"] + 0.005,
        "S2 B3 CC-BY ref",
        fontsize=7,
        color="#666",
        transform=axS.get_yaxis_transform(),
    )
    axS.axhline(REF["S2 PRUE-B7 (CC-BY) ref"], color="#888", linestyle="--", linewidth=0.8)
    axS.text(
        0.02,
        REF["S2 PRUE-B7 (CC-BY) ref"] + 0.005,
        "S2 B7 CC-BY ref",
        fontsize=7,
        color="#666",
        transform=axS.get_yaxis_transform(),
    )
    axS.axhline(REF["S2 PRUE-B7 full ref"], color="#a44", linestyle=":", linewidth=0.8)
    axS.text(
        0.02,
        REF["S2 PRUE-B7 full ref"] + 0.005,
        "S2 B7 full ref",
        fontsize=7,
        color="#a44",
        transform=axS.get_yaxis_transform(),
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "aug_ablation.pdf", bbox_inches="tight")
    print(f"wrote {FIGS / 'aug_ablation.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries",
        choices=("heldout11", "heldout9"),
        default="heldout11",
        help="Country set driving the figure bars. CSVs for BOTH sets are always emitted.",
    )
    args = parser.parse_args()

    # Always emit both tables.
    p11, s11 = _build_tables(HELDOUT_11, "heldout11")
    p9, s9 = _build_tables(HELDOUT_9, "heldout9")

    if args.countries == "heldout11":
        _draw(p11, s11, "11-country")
    else:
        _draw(p9, s9, "9-country")


if __name__ == "__main__":
    main()
