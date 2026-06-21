"""Generate ``paper/figs/heldout_per_country.tex`` (``tab:heldout_pc``).

Per-country pixel IoU + Planet Obj F1 (WS+TTA) on the 11 held-out
countries. S2 column is our FTW-PRUE+ (B3) full model at the upsampled-512
(resize_factor=2) eval
(``logs/postproc_ablation/s2_b3_augmax_full_upsampled_ws_tta.csv``); Planet
column is our best B3 PRUE+ full + WS + D4 TTA
(``logs/repro_eval/pp_ws_tta.csv``).

Kenya's labels are presence-only (background untrusted): its supervised
pixel/object metrics are not comparable, so the kenya row shows dashes and
the macro row averages the 10 dense-label countries only.

Run::

    uv run python paper/scripts/make_heldout_per_country.py
"""

from pathlib import Path

import pandas as pd
from _aggregate import HELDOUT_10_DENSE, HELDOUT_11

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "heldout_per_country.tex"

S2_CSV = REPO / "logs" / "postproc_ablation" / "s2_b3_augmax_full_upsampled_ws_tta.csv"
# Released B3-full checkpoint (retrained Jun 2026, epoch 92): reproduction eval.
PL_CSV = REPO / "logs" / "repro_eval" / "pp_ws_tta.csv"

PRESENCE_ONLY = {"kenya"}


def main() -> None:
    s2 = pd.read_csv(S2_CSV).set_index("country")
    pl = pd.read_csv(PL_CSV).set_index("country")

    # Planet eval (released ckpt) excludes presence-only kenya by construction;
    # only the dense-10 rows are required there. S2 ref still carries all 11.
    missing_s2 = set(HELDOUT_10_DENSE) - set(s2.index)
    missing_pl = set(HELDOUT_10_DENSE) - set(pl.index)
    if missing_s2 or missing_pl:
        raise RuntimeError(f"missing rows: s2={missing_s2}, planet={missing_pl}")

    rows: list[str] = []
    rows.append(r"\footnotesize")
    rows.append(r"\setlength{\tabcolsep}{4pt}")
    rows.append(r"\begin{tabular}{@{}l c c c c@{}}")
    rows.append(r"\toprule")
    rows.append(
        r"Country & \makecell{FTW-PRUE+\\(B3)\\IoU} & "
        r"\makecell{FTP-PRUE+\\(B3)\\IoU} & \makecell{$\Delta$\\IoU} & "
        r"\makecell{FTP-PRUE+\\Obj F1\\(WS+TTA)} \\"
    )
    rows.append(r"\midrule")

    wins = 0
    s2_sum = 0.0
    pl_sum = 0.0
    delta_sum = 0.0
    f1_sum = 0.0
    for c in HELDOUT_11:
        c_pretty = c.replace("_", " ").title()
        if c in PRESENCE_ONLY:
            rows.append(rf"{c_pretty}$^{{\dagger}}$ & --- & --- & --- & --- \\")
            continue
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
            s2_cell = f"{s2_iou * 100:.1f}"
            pl_cell = rf"\textbf{{{pl_iou * 100:.1f}}}"
        else:
            s2_cell = rf"\textbf{{{s2_iou * 100:.1f}}}"
            pl_cell = f"{pl_iou * 100:.1f}"
        rows.append(
            f"{c_pretty} & {s2_cell} & {pl_cell} & {delta * 100:+.1f} & {pl_f1 * 100:.1f} \\\\"
        )

    n = len(HELDOUT_10_DENSE)
    rows.append(r"\midrule")
    rows.append(
        f"Macro ({n}, dense) & {s2_sum / n * 100:.1f} "
        rf"& \textbf{{{pl_sum / n * 100:.1f}}} & {delta_sum / n * 100:+.1f} "
        rf"& \textbf{{{f1_sum / n * 100:.1f}}} \\"
    )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append("")
    rows.append(
        r"\vspace{2pt}\noindent{\scriptsize $^{\dagger}$Kenya labels are "
        r"presence-only (background untrusted); supervised pixel/object "
        r"metrics are omitted (see \Cref{sec:limitations}).}"
    )
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")
    print(f"Planet wins {wins}/{n} (dense-label countries)")


if __name__ == "__main__":
    main()
