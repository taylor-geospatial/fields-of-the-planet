"""Generate ``paper/figs/full_data_compare.tex`` (``tab:full_data``).

Dense-label held-out macro-average (HELDOUT_10_DENSE) comparing the released
FTW S2 PRUE checkpoints (numbers as published in~\\cite{muhawenayo2026prue})
to our PRUE+ recipe on Planet.

Kenya is excluded by construction twice over: its labels are presence-only
(see ``_aggregate.HELDOUT_10_DENSE``) and the FTW ``full_data`` protocol's
22-country test split does not include it, so ``logs/fulldata_eval/*.csv``
has no kenya rows.

Run::

    uv run python paper/scripts/full_data_table.py
"""

from pathlib import Path

from _aggregate import HELDOUT_10_DENSE, macro_avg

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "full_data_compare.tex"
SRC = REPO / "logs" / "fulldata_eval"
REPRO = REPO / "logs" / "repro_eval"  # released B3-full checkpoint reproduction eval

# Released FTW PRUE numbers (hand-copied from muhawenayo2026prue; rows kept as
# reference). These are the published values on the FTW full_data test split,
# not the held-out macro -- we keep them unchanged for orientation.
# (model, backbone, pix IoU, obj F1)
RELEASED_FULL = [
    ("FTW-PRUE", "B3", 0.74, 0.43),
    ("FTW-PRUE", "B5", 0.75, 0.46),
    ("FTW-PRUE", "B7", 0.76, 0.47),
]

# (model, backbone, csv path); the B3 row is the released checkpoint (repro eval).
OURS = [
    ("FTP-PRUE+", "B3", REPRO / "pp_ws_tta.csv"),
]


def main() -> None:
    rows: list[str] = []
    rows.append(r"\footnotesize")
    rows.append(r"\setlength{\tabcolsep}{3pt}")
    rows.append(r"\begin{tabular}{@{}l@{\hspace{4pt}}l@{\hspace{4pt}}c@{\hspace{4pt}}c@{}}")
    rows.append(r"\toprule")
    rows.append(r"Model & Backbone & Pix IoU & Obj F1 \\")
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{4}{l}{\textit{FTW-PRUE, released by \cite{muhawenayo2026prue}}} \\"
    )
    for model, backbone, iou, f1 in RELEASED_FULL:
        bold = backbone == "B7"
        f1s = rf"\textbf{{{f1 * 100:.1f}}}" if bold else f"{f1 * 100:.1f}"
        rows.append(f"{model} & {backbone} & {iou * 100:.1f} & {f1s} \\\\")
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{4}{l}{\textit{Ours --- FTP-PRUE+ (10-country dense held-out macro)}} \\"
    )
    for model, backbone, csv_path in OURS:
        agg = macro_avg(csv_path, HELDOUT_10_DENSE)
        nc, ne = int(agg["n_countries"]), int(agg["n_expected"])
        if nc != ne:
            raise RuntimeError(f"{csv_path}: macro over {nc}/{ne} countries")
        rows.append(
            f"{model} & {backbone} & "
            f"{agg['pixel_level_iou'] * 100:.1f} & {agg['object_ws_f1'] * 100:.1f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
