"""Generate ``paper/figs/full_data_compare.tex`` (``tab:full_data``).

Dense-label held-out macro-average (HELDOUT_10_DENSE) comparing the released
FTW S2 PRUE checkpoints (numbers as published in~\\cite{muhawenayo2026prue})
to our augmax recipe on Planet.

Kenya is excluded by construction twice over: its labels are presence-only
(see ``_aggregate.HELDOUT_10_DENSE``) and the FTW ``full_data`` protocol's
22-country test split does not include it, so ``logs/fulldata_eval/*.csv``
has no kenya rows.

Run::

    uv run python paper/scripts/make_full_data_compare.py
"""

from pathlib import Path

from _aggregate import HELDOUT_10_DENSE, macro_avg

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "full_data_compare.tex"
SRC = REPO / "logs" / "fulldata_eval"

# Released FTW PRUE numbers (hand-copied from muhawenayo2026prue; rows kept as
# reference). These are the published values on the FTW full_data test split,
# not the held-out macro -- we keep them unchanged for orientation.
RELEASED_FULL = [
    ("PRUE-B3 (S2, full)", "24/25", 0.74, 0.43),
    ("PRUE-B5 (S2, full)", "24/25", 0.75, 0.46),
    ("PRUE-B7 (S2, full)", "24/25", 0.76, 0.47),
]
RELEASED_CCBY = [
    ("PRUE-B3 (S2, CC-BY)", "14", 0.76, 0.39),
    ("PRUE-B5 (S2, CC-BY)", "14", 0.76, 0.41),
    ("PRUE-B7 (S2, CC-BY)", "14", 0.77, 0.44),
]

OURS = [
    ("PRUE-FTP-B3 \\emph{augmax}", "14", "planet_b3_augmax_ccby_ws_tta.csv"),
    ("PRUE-FTP-B3 \\emph{augmax}", "24/25", "planet_b3_augmax_full_ws_tta.csv"),
    ("PRUE-FTP-B7 \\emph{augmax}", "14", "planet_b7_augmax_ccby_ws_tta.csv"),
]


def main() -> None:
    rows: list[str] = []
    rows.append(r"\footnotesize")
    rows.append(r"\setlength{\tabcolsep}{3pt}")
    rows.append(r"\begin{tabular}{@{}l@{\hspace{4pt}}l@{\hspace{4pt}}c@{\hspace{4pt}}c@{}}")
    rows.append(r"\toprule")
    rows.append(r"Model & Train set & Pix IoU & Obj F1 \\")
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{4}{l}{\textit{S2 PRUE CC-BY-NC (released by \cite{muhawenayo2026prue})}} \\"
    )
    for name, train, iou, f1 in RELEASED_FULL:
        bold = name.startswith("PRUE-B7")
        f1s = rf"\textbf{{{f1:.2f}}}" if bold else f"{f1:.2f}"
        rows.append(f"{name} & {train} & {iou:.2f} & {f1s} \\\\")
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{4}{l}{\textit{S2 PRUE CC-BY (released by \cite{muhawenayo2026prue})}} \\"
    )
    for name, train, iou, f1 in RELEASED_CCBY:
        bold_iou = name.startswith("PRUE-B7")
        ious = rf"\textbf{{{iou:.2f}}}" if bold_iou else f"{iou:.2f}"
        rows.append(f"{name} & {train} & {ious} & {f1:.2f} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{l}{\textit{Ours (PRUE-FTP, 10-country dense held-out macro)}} \\")
    for name, train, csv_name in OURS:
        agg = macro_avg(SRC / csv_name, HELDOUT_10_DENSE)
        nc, ne = int(agg["n_countries"]), int(agg["n_expected"])
        if nc != ne:
            raise RuntimeError(f"{csv_name}: macro over {nc}/{ne} countries")
        rows.append(
            f"{name} & {train} & {agg['pixel_level_iou']:.2f} & {agg['object_ws_f1']:.2f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
