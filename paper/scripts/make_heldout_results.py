"""Generate ``paper/figs/heldout_results.tex`` (``tab:heldout``).

11-country held-out macro-average over four post-processing combos
(WS x TTA) for each (imagery, backbone, train-split) configuration.

Sources: ``logs/postproc_ablation/<stem>_<combo>.csv`` (per-country rows for
all 11 dense-label held-out countries; ``object_ws_f1`` when WS is on,
``object_pix_f1`` when off; pixel IoU comes from the WS+TTA file).

Run::

    uv run python paper/scripts/make_heldout_results.py
"""

from pathlib import Path

from _aggregate import HELDOUT_11, macro_avg

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "heldout_results.tex"
SRC = REPO / "logs" / "postproc_ablation"

# (display label, csv stem) — same row order as the previous table.
CONFIGS_OURS_PLANET = [
    ("B3 (CC-BY)", "planet_b3_augmax_ccby"),
    ("B7 (CC-BY)", "planet_b7_augmax_ccby"),
    ("B3 (full, 24/25)", "planet_b3_augmax_full"),
]
CONFIGS_OURS_S2 = [
    ("B3 (CC-BY)", "s2_b3_augmax_ccby"),
    ("B7 (CC-BY)", "s2_b7_augmax_ccby"),
    ("B3 (full, 24/25)", "s2_b3_augmax_full"),
    ("B7 (full, 24/25)", "s2_b7_augmax_full"),
]

COMBOS = (
    ("nows_notta", False),
    ("nows_tta", False),
    ("ws_notta", True),
    ("ws_tta", True),
)


def _row(stem: str) -> tuple[list[float], float, int, int]:
    """Return (obj_f1 across 4 combos, pixel IoU from ws_tta, n_countries, n_expected)."""
    vals: list[float] = []
    n_countries = 0
    n_expected = 0
    pix_iou = 0.0
    for combo, ws in COMBOS:
        csv = SRC / f"{stem}_{combo}.csv"
        agg = macro_avg(csv, HELDOUT_11)
        key = "object_ws_f1" if ws else "object_pix_f1"
        vals.append(agg[key])
        if combo == "ws_tta":
            pix_iou = float(agg["pixel_level_iou"])
            n_countries = int(agg["n_countries"])
            n_expected = int(agg["n_expected"])
    return vals, pix_iou, n_countries, n_expected


def _fmt_row(label: str, imagery: str, vals: list[float], pix_iou: float) -> str:
    best = max(vals)
    cells = []
    for v in vals:
        s = f"{v:.3f}"
        if v == best:
            s = rf"\textbf{{{s}}}"
        cells.append(s)
    best_iou_flag = ""  # bolding for pix iou applied across-table elsewhere if needed
    return (
        f"{imagery} & {label}                       "
        f"& {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} "
        f"& {pix_iou:.3f}{best_iou_flag} \\\\"
    )


def main() -> None:
    rows: list[str] = []

    rows.append(r"\begin{tabular}{llccccc}")
    rows.append(r"\toprule")
    rows.append(
        r"& & \multicolumn{4}{c}{Postprocessing combo --- Obj F1 (11-country held-out)} & \\"
    )
    rows.append(r"\cmidrule(lr){3-6}")
    rows.append(
        r"Imagery & Recipe (CC-BY split, 14-country train unless noted) "
        r"& no-WS / no-TTA & no-WS / TTA & WS / no-TTA & \textbf{WS + TTA} & Pix IoU \\"
    )
    rows.append(r"\midrule")
    # Released FTW S2 PRUE reference numbers (hand-copied from kerner2024ftw,
    # on the FTW full_data test split — not directly comparable to our
    # 11-country held-out macro; kept for orientation).
    rows.append(
        r"S2 & FTW PRUE-B3 (CC-BY) \cite{kerner2024ftw} & --- & --- & --- & 0.39$^\ddag$ & 0.76$^\ddag$ \\"
    )
    rows.append(
        r"S2 & FTW PRUE-B7 (CC-BY) \cite{kerner2024ftw} & --- & --- & --- & 0.44$^\ddag$ & 0.77$^\ddag$ \\"
    )
    rows.append(
        r"S2 & FTW PRUE-B7 (full, 24/25) \cite{kerner2024ftw} & --- & --- & --- & 0.47$^\ddag$ & 0.76$^\ddag$ \\"
    )
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{7}{l}{\textit{Ours --- PRUE-FTP \textbf{augmax} (this report)}} \\"
    )
    # Find best WS+TTA Obj F1 across our Planet rows for bolding.
    planet_vals: list[tuple[list[float], float, int, int]] = [
        _row(stem) for _, stem in CONFIGS_OURS_PLANET
    ]
    planet_best_wstta = max(v[0][3] for v in planet_vals)
    planet_best_iou = max(v[1] for v in planet_vals)
    for (label, _), (vals, pix_iou, nc, ne) in zip(CONFIGS_OURS_PLANET, planet_vals):
        if nc != ne:
            raise RuntimeError(
                f"Planet config {label}: macro over {nc}/{ne} countries; "
                f"expected all 11 of HELDOUT_11."
            )
        cells: list[str] = []
        for i, v in enumerate(vals):
            s = f"{v:.3f}"
            if (i == 3 and v == planet_best_wstta) or v == max(vals):
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        iou_s = f"{pix_iou:.3f}"
        if pix_iou == planet_best_iou:
            iou_s = rf"\textbf{{{iou_s}}}"
        rows.append(
            f"Planet & {label}                       "
            f"& {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {iou_s} \\\\"
        )
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{7}{l}{\textit{Ours --- Sentinel-2 \textbf{augmax} (this report)}} \\"
    )
    s2_vals = [_row(stem) for _, stem in CONFIGS_OURS_S2]
    for (label, _), (vals, pix_iou, nc, ne) in zip(CONFIGS_OURS_S2, s2_vals):
        if nc != ne:
            raise RuntimeError(
                f"S2 config {label}: macro over {nc}/{ne} countries; expected all 11 of HELDOUT_11."
            )
        cells = []
        best = max(vals)
        for v in vals:
            s = f"{v:.3f}"
            if v == best:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        rows.append(
            f"S2 & {label}                       "
            f"& {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {pix_iou:.3f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append("")
    rows.append(r"\vspace{0.4em}")
    rows.append(
        r"\noindent\footnotesize $^\ddag$Reported by~\cite{muhawenayo2026prue} on the FTW "
        r"\texttt{full\_data} test split (includes test patches from the 14 training "
        r"countries). Our numbers are macro-averaged over all 11 held-out "
        r"countries (belgium, cambodia, croatia, germany, kenya, latvia, lithuania, "
        r"portugal, slovenia, south\_africa, sweden). Kenya is presence-only, so "
        r"pixel IoU there is shown only for protocol continuity; see~\Cref{sec:limitations}. "
        r"The \emph{full} rows of our recipe use the 24-country / 25-region train "
        r"split for the full-data protocol."
    )
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
