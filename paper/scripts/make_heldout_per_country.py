"""Generate ``paper/figs/heldout_per_country.tex`` (``tab:heldout_pc``).

Per-country pixel IoU + Planet Obj F1 (WS+TTA) on the 11 dense-label
held-out countries. S2 column is the released FTW S2 PRUE-B3 CC-BY
checkpoint as evaluated in ``logs/ftw_official_ccby/s2_ccby_per_country.csv``;
Planet column is our best B3 \\emph{augmax} full + WS + D4 TTA
(``logs/postproc_ablation/planet_b3_augmax_full_ws_tta.csv``).

Run::

    uv run python paper/scripts/make_heldout_per_country.py
"""

from pathlib import Path

import pandas as pd

from _aggregate import HELDOUT_11

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "heldout_per_country.tex"

S2_CSV = REPO / "logs" / "ftw_official_ccby" / "s2_ccby_per_country.csv"
PL_CSV = REPO / "logs" / "postproc_ablation" / "planet_b3_augmax_full_ws_tta.csv"


def main() -> None:
    s2 = pd.read_csv(S2_CSV).set_index("country")
    pl = pd.read_csv(PL_CSV).set_index("country")

    missing_s2 = set(HELDOUT_11) - set(s2.index)
    missing_pl = set(HELDOUT_11) - set(pl.index)
    if missing_s2 or missing_pl:
        raise RuntimeError(f"missing rows: s2={missing_s2}, planet={missing_pl}")

    rows: list[str] = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append(
        r"Country & S2 CCBY IoU & Planet B3 \textbf{augmax} full IoU "
        r"& $\Delta$ IoU & Planet Obj F1 (WS+TTA) \\"
    )
    rows.append(r"\midrule")

    wins = 0
    s2_sum = 0.0
    pl_sum = 0.0
    delta_sum = 0.0
    f1_sum = 0.0
    for c in HELDOUT_11:
        s2_iou = float(s2.loc[c, "pixel_level_iou"])
        pl_iou = float(pl.loc[c, "pixel_level_iou"])
        pl_f1 = float(pl.loc[c, "object_ws_f1"])
        delta = pl_iou - s2_iou
        s2_sum += s2_iou
        pl_sum += pl_iou
        delta_sum += delta
        f1_sum += pl_f1
        if pl_iou > s2_iou:
            wins += 1
            s2_cell = f"{s2_iou:.3f}"
            pl_cell = rf"\textbf{{{pl_iou:.3f}}}"
        else:
            s2_cell = rf"\textbf{{{s2_iou:.3f}}}"
            pl_cell = f"{pl_iou:.3f}"
        c_pretty = c.replace("_", " ")
        rows.append(
            f"{c_pretty} & {s2_cell} & {pl_cell} & {delta:+.3f} & {pl_f1:.3f} \\\\"
        )

    n = len(HELDOUT_11)
    rows.append(r"\midrule")
    rows.append(
        f"Macro mean (11, incl K+P) & {s2_sum / n:.3f} "
        rf"& \textbf{{{pl_sum / n:.3f}}} & {delta_sum / n:+.3f} "
        rf"& \textbf{{{f1_sum / n:.3f}}} \\"
    )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")
    print(f"Planet wins {wins}/{n}")


if __name__ == "__main__":
    main()
