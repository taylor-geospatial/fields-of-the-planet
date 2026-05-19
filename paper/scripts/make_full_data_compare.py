"""Generate ``paper/figs/full_data_compare.tex`` (``tab:full_data``).

11-country held-out macro-average comparing the released FTW S2 PRUE
checkpoints (numbers as published in~\\cite{kerner2024ftw}) to our augmax
recipe on Planet and S2.

The ``logs/fulldata_eval/*.csv`` files are missing kenya at the time of
writing (n=10/11). We surface the gap in the table-note rather than silently
averaging over fewer countries.

Run::

    uv run python paper/scripts/make_full_data_compare.py
"""

from pathlib import Path

from _aggregate import HELDOUT_11, macro_avg

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "full_data_compare.tex"
SRC = REPO / "logs" / "fulldata_eval"

# Released FTW PRUE numbers (hand-copied from kerner2024ftw; rows kept as
# reference). These are the published values on the FTW full_data test split,
# not the 11-country held-out macro -- we keep them unchanged for orientation.
RELEASED = [
    ("S2 PRUE-B3 (full)", "all 25", 0.74, 0.43),
    ("S2 PRUE-B5 (full)", "all 25", 0.75, 0.46),
    ("S2 PRUE-B7 (full)", "all 25", 0.76, 0.47),
    ("S2 PRUE-B3 (CC-BY)", "14", 0.76, 0.39),
    ("S2 PRUE-B5 (CC-BY)", "14", 0.76, 0.41),
    ("S2 PRUE-B7 (CC-BY)", "14", 0.77, 0.44),
]

OURS = [
    ("FTW-Planet B3 \\emph{augmax}", "14", "planet_b3_augmax_ccby_ws_tta.csv"),
    ("FTW-Planet B3 \\emph{augmax}", "all 25", "planet_b3_augmax_full_ws_tta.csv"),
    ("FTW-Planet B7 \\emph{augmax}", "14", "planet_b7_augmax_ccby_ws_tta.csv"),
]


def main() -> None:
    rows: list[str] = []
    rows.append(r"\setlength{\tabcolsep}{4pt}")
    rows.append(r"\begin{tabular}{llcc}")
    rows.append(r"\toprule")
    rows.append(r"Model & Train countries & Pix IoU & Obj F1 \\")
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{4}{l}{\textit{S2 PRUE CC-BY-NC (released by \cite{kerner2024ftw})}} \\"
    )
    for name, train, iou, f1 in RELEASED[:3]:
        bold = name.endswith("B7 (full)")
        f1s = rf"\textbf{{{f1:.2f}}}" if bold else f"{f1:.2f}"
        rows.append(f"{name} & {train} & {iou:.2f} & {f1s} \\\\")
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{4}{l}{\textit{S2 PRUE CC-BY (released by \cite{kerner2024ftw})}} \\"
    )
    for name, train, iou, f1 in RELEASED[3:]:
        bold_iou = "B7" in name and "(CC-BY)" in name
        ious = rf"\textbf{{{iou:.2f}}}" if bold_iou else f"{iou:.2f}"
        rows.append(f"{name} & {train} & {ious} & {f1:.2f} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{4}{l}{\textit{Ours (FTW-Planet, 11-country held-out macro)}} \\")
    n_seen: list[tuple[int, int]] = []
    for name, train, csv_name in OURS:
        agg = macro_avg(SRC / csv_name, HELDOUT_11)
        n_seen.append((int(agg["n_countries"]), int(agg["n_expected"])))
        rows.append(
            f"{name} & {train} & {agg['pixel_level_iou']:.2f} & {agg['object_ws_f1']:.2f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")

    # Surface the gap.
    nc_set = {nc for nc, _ in n_seen}
    if nc_set != {11}:
        # Build a missing-country list from the first CSV we touch
        from _aggregate import load_and_filter

        sub = load_and_filter(SRC / OURS[0][2], HELDOUT_11)
        missing = sorted(set(HELDOUT_11) - set(sub["country"].tolist()))
        miss_pretty = ", ".join(c.replace("_", " ") for c in missing)
        rows.append("")
        rows.append(r"\vspace{0.3em}")
        rows.append(
            rf"\noindent\footnotesize $^*$Our numbers are macro-averaged over "
            rf"{max(nc for nc, _ in n_seen)} of the 11 dense-label held-out "
            rf"countries. {miss_pretty.capitalize()} pending re-evaluation under "
            rf"the \texttt{{full\_data}} protocol and will be folded in once the "
            rf"missing CSV row is regenerated. Released FTW PRUE rows are the "
            rf"published values on the FTW \texttt{{full\_data}} 22-country test "
            rf"split (not directly comparable to our 11-country held-out macro; "
            rf"kept for orientation)."
        )

    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")
    print("n_seen per ours row (n, expected):", n_seen)


if __name__ == "__main__":
    main()
