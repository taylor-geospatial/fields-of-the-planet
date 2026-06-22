"""Generate ``paper/figs/heldout_results.tex`` (``tab:heldout``).

Dense-label held-out macro-average (HELDOUT_10_DENSE: the 11 held-out
countries minus presence-only kenya) over four post-processing combos
(WS x TTA) for each (imagery, backbone, train-split) configuration.

Sources: ``logs/postproc_ablation/<stem>_<combo>.csv`` (per-country rows;
``object_ws_f1`` when WS is on, ``object_pix_f1`` when off; pixel IoU comes
from the WS+TTA file).

Run::

    uv run python paper/scripts/heldout_results_table.py
"""

from pathlib import Path

from _aggregate import HELDOUT_10_DENSE, macro_avg

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "heldout_results.tex"
SRC = REPO / "logs" / "postproc_ablation"
# Released B3-full checkpoint (retrained Jun 2026): its per-combo CSVs live in
# logs/repro_eval/pp_<combo>.csv instead of the original postproc_ablation run.
REPRO = REPO / "logs" / "repro_eval"
REPRO_STEM = "planet_b3_augmax_full"

# (model, backbone, csv stem). Full-data split only.
CONFIGS_OURS_PLANET = [
    ("FTP-PRUE+", "B3", "planet_b3_augmax_full"),
]
# S2 stems use the upsampled-512 (resize_factor=2) postproc sweep, matching the
# canonical PRUE eval used everywhere else for Sentinel-2 (job 112416).
CONFIGS_OURS_S2 = [
    ("FTW-PRUE+", "B3", "s2_b3_augmax_full_upsampled"),
    ("FTW-PRUE+", "B7", "s2_b7_augmax_full_upsampled"),
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
        csv = REPRO / f"pp_{combo}.csv" if stem == REPRO_STEM else SRC / f"{stem}_{combo}.csv"
        agg = macro_avg(csv, HELDOUT_10_DENSE)
        key = "object_ws_f1" if ws else "object_pix_f1"
        vals.append(agg[key])
        if combo == "ws_tta":
            pix_iou = float(agg["pixel_level_iou"])
            n_countries = int(agg["n_countries"])
            n_expected = int(agg["n_expected"])
    return vals, pix_iou, n_countries, n_expected


def _cells(vals: list[float]) -> list[str]:
    best = max(vals)
    out = []
    for v in vals:
        s = f"{v * 100:.1f}"
        if v == best:
            s = rf"\textbf{{{s}}}"
        out.append(s)
    return out


def main() -> None:
    rows: list[str] = []
    rows.append(r"\footnotesize")
    rows.append(r"\setlength{\tabcolsep}{3pt}")
    rows.append(r"\renewcommand{\arraystretch}{1.05}")
    rows.append(r"\begin{tabular}{@{}l l ccc c c@{}}")
    rows.append(r"\toprule")
    rows.append(r"& & \multicolumn{4}{c}{Obj F1 (10-country dense held-out)} & \\")
    rows.append(r"\cmidrule(lr){3-6}")
    rows.append(
        r"Model & Backbone & "
        r"\makecell{no-WS\\no-TTA} & \makecell{no-WS\\TTA} & "
        r"\makecell{WS\\no-TTA} & \makecell{\textbf{WS}\\\textbf{+TTA}} & "
        r"\makecell{Pix\\IoU} \\"
    )
    rows.append(r"\midrule")
    # Released PRUE reference number (constant from muhawenayo2026prue, on the
    # FTW full_data test split — not directly comparable to our held-out macro;
    # kept for orientation).
    rows.append(
        r"FTW-PRUE & B7~\cite{muhawenayo2026prue}  & --- & --- & --- "
        r"& 47.0$^\ddag$ & 76.0$^\ddag$ \\"
    )
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{@{}l}{\textit{Ours --- FTP-PRUE+}} \\")

    planet_vals = [_row(stem) for *_, stem in CONFIGS_OURS_PLANET]
    planet_best_iou = max(v[1] for v in planet_vals)
    for (model, backbone, _), (vals, pix_iou, nc, ne) in zip(CONFIGS_OURS_PLANET, planet_vals):
        if nc != ne:
            raise RuntimeError(
                f"Planet config {model} {backbone}: macro over {nc}/{ne} countries; "
                f"expected all {len(HELDOUT_10_DENSE)} of HELDOUT_10_DENSE."
            )
        cells = _cells(vals)
        iou_s = f"{pix_iou * 100:.1f}"
        if pix_iou == planet_best_iou:
            iou_s = rf"\textbf{{{iou_s}}}"
        rows.append(
            f"{model} & {backbone} & "
            f"{cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {iou_s} \\\\"
        )
    rows.append(r"\midrule")
    rows.append(
        r"\multicolumn{7}{@{}l}{\textit{FTW-PRUE baselines re-trained with our \textbf{PRUE+} recipe}} \\"
    )
    s2_vals = [_row(stem) for *_, stem in CONFIGS_OURS_S2]
    for (model, backbone, _), (vals, pix_iou, nc, ne) in zip(CONFIGS_OURS_S2, s2_vals):
        if nc != ne:
            raise RuntimeError(
                f"S2 config {model} {backbone}: macro over {nc}/{ne} countries; "
                f"expected all {len(HELDOUT_10_DENSE)} of HELDOUT_10_DENSE."
            )
        cells = _cells(vals)
        rows.append(
            f"{model} & {backbone} & "
            f"{cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {pix_iou * 100:.1f} \\\\"
        )
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append("")
    rows.append(
        r"\vspace{2pt}\noindent{\scriptsize $^\ddag$ Released PRUE checkpoint numbers; "
        r"we did not re-evaluate WS/TTA combos for them. Macros exclude presence-only "
        r"kenya (see \Cref{sec:limitations}); portugal is retained.}"
    )
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT}")
    for (model, backbone, _), (vals, pix_iou, _, _) in zip(
        CONFIGS_OURS_PLANET + CONFIGS_OURS_S2, planet_vals + s2_vals
    ):
        print(f"  {model} {backbone}: ws_tta={vals[3]:.3f} pix_iou={pix_iou:.3f}")


if __name__ == "__main__":
    main()
