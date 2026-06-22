"""Generate ``paper/figs/polygon_metrics.tex`` (``tab:polygon_metrics``).

Dense-label held-out macro-average (HELDOUT_10_DENSE) of polygon metrics, the
secondary pixel IoU, and PQ by GT field-area bin, for each method:

* DelineateAnything / DelineateAnything-S (YOLO11x / YOLO11n), zero-shot, on
  both Planet (3 m) and S2 (10 m); sensor is folded into the method name.
* FTW-PRUE+ B3/B7 (Sentinel-2) and FTP-PRUE+ B3/B7 (PlanetScope, ours).

Each row pulls polygon metrics + |dN|/N from its polygon CSV, the boundary
chamfer at its native grid, pixel IoU from its post-processing CSV, and PQ by
area bin from its ``*.bins.csv``. Sources are listed per row below.

Run::

    uv run python paper/scripts/polygon_metrics_table.py
"""

from pathlib import Path

import pandas as pd
from _aggregate import HELDOUT_10_DENSE, load_and_filter

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "polygon_metrics.tex"
PM = REPO / "logs" / "polygon_metrics"
AREA = REPO / "logs" / "area_bins"
PP = REPO / "logs" / "postproc_ablation"
REPRO = REPO / "logs" / "repro_eval"

_DA = r"~\cite{lavreniuk2025delineate}"
_PRUE = r"~\cite{muhawenayo2026prue}"

# Each row: display name, backbone, polygon-metrics CSV, area-bins CSV,
# pixel-IoU CSV (pixel_level_iou col), boundary CSV (None = use polygon CSV),
# bold-backbone, midrule-before.
ROWS = [
    (
        rf"DelineateAnything (S2)$^{{*}}${_DA}",
        "YOLO11x",
        PM / "delineate_x_s2.csv",
        PM / "delineate_x_s2.csv.bins.csv",
        PM / "delineate_x_s2.csv",
        None,
        False,
        False,
    ),
    (
        rf"DelineateAnything-S (S2)$^{{*}}${_DA}",
        "YOLO11n",
        PM / "delineate_s_s2.csv",
        PM / "delineate_s_s2.csv.bins.csv",
        PM / "delineate_s_s2.csv",
        None,
        False,
        False,
    ),
    (
        rf"DelineateAnything (Planet)$^{{*}}${_DA}",
        "YOLO11x",
        PM / "delineate_x_planet.csv",
        PM / "delineate_x_planet.csv.bins.csv",
        PM / "delineate_x_planet.csv",
        None,
        False,
        False,
    ),
    (
        rf"DelineateAnything-S (Planet)$^{{*}}${_DA}",
        "YOLO11n",
        PM / "delineate_s_planet.csv",
        PM / "delineate_s_planet.csv.bins.csv",
        PM / "delineate_s_planet.csv",
        None,
        False,
        False,
    ),
    (
        rf"FTW-PRUE+ (S2){_PRUE}",
        "B3",
        PM / "s2_b3_augmax_full_upsampled_22.csv",
        AREA / "s2_b3.csv.bins.csv",
        PP / "s2_b3_augmax_full_upsampled_ws_tta.csv",
        PM / "s2_b3_augmax_full_native256.csv",
        False,
        True,
    ),
    (
        rf"FTW-PRUE+ (S2){_PRUE}",
        "B7",
        PM / "s2_upsampled_b7_augmax_full_22.csv",
        AREA / "s2_b7.csv.bins.csv",
        PP / "s2_b7_augmax_full_upsampled_ws_tta.csv",
        PM / "s2_b7_augmax_full_native256.csv",
        False,
        False,
    ),
    (
        r"\textbf{FTP-PRUE+ (Planet, ours)}",
        "B3",
        REPRO / "polygon_metrics.csv",
        AREA / "planet_b3.csv.bins.csv",
        REPRO / "pp_ws_tta.csv",
        None,
        True,
        True,
    ),
    (
        r"\textbf{FTP-PRUE+ (Planet, ours)}",
        "B7",
        AREA / "planet_b7.csv",
        AREA / "planet_b7.csv.bins.csv",
        PP / "planet_b7_augmax_full_ws_tta.csv",
        None,
        True,
        False,
    ),
]

POLY_COLS = ("pq", "pq_sq", "pq_rq", "ap_5_95")
BND_COLS = ("boundary_error_m_mean", "boundary_error_m_p95")
AREA_BINS = ("small", "medium", "large")


def _macro(csv: Path, col: str) -> float:
    sub = load_and_filter(csv, HELDOUT_10_DENSE)
    if len(sub) != len(HELDOUT_10_DENSE):
        raise RuntimeError(f"{csv}: macro over {len(sub)}/{len(HELDOUT_10_DENSE)} countries")
    return float(sub[col].mean(skipna=True))


def _norm_count(csv: Path) -> float:
    sub = load_and_filter(csv, HELDOUT_10_DENSE)
    return float((sub["polygon_count_delta_mean"] / sub["n_gt_mean"]).mean())


def _area_pq(csv: Path) -> dict[str, float]:
    d = pd.read_csv(csv).set_index("bin")
    return {b: float(d.loc[b, "pq"]) for b in AREA_BINS}


def main() -> None:
    aggs: list[dict[str, float]] = []
    for _, _, poly, area_csv, pix_csv, bnd_csv, _, _ in ROWS:
        agg = {c: _macro(poly, c) for c in POLY_COLS}
        agg["dN_norm"] = _norm_count(poly)
        bnd = bnd_csv if bnd_csv is not None else poly
        for c in BND_COLS:
            agg[c] = _macro(bnd, c)
        agg["pixel_iou"] = _macro(pix_csv, "pixel_level_iou")
        for b, v in _area_pq(area_csv).items():
            agg[f"pq_{b}"] = v
        aggs.append(agg)

    area_cols = tuple(f"pq_{b}" for b in AREA_BINS)
    metric_cols = (*POLY_COLS, "dN_norm", *BND_COLS, "pixel_iou", *area_cols)
    higher_better = {"pq", "pq_sq", "pq_rq", "ap_5_95", "pixel_iou", *area_cols}
    best = {
        c: (max if c in higher_better else min)(a[c] for a in aggs if a[c] == a[c])
        for c in metric_cols
    }

    def cell(v: float, c: str, dec: int, scale: float) -> str:
        if v != v:
            return "--"
        s = f"{v * scale:.{dec}f}"
        return rf"\textbf{{{s}}}" if abs(v - best[c]) < 1e-9 else s

    def row_line(name: str, bb: str, bold_bb: bool, agg: dict[str, float]) -> str:
        bb_s = rf"\textbf{{{bb}}}" if bold_bb else bb
        return (
            f"{name} & {bb_s} & "
            f"{cell(agg['pq'], 'pq', 1, 100)} & {cell(agg['pq_sq'], 'pq_sq', 1, 100)} & "
            f"{cell(agg['pq_rq'], 'pq_rq', 1, 100)} & {cell(agg['ap_5_95'], 'ap_5_95', 1, 100)} & "
            f"{cell(agg['dN_norm'], 'dN_norm', 2, 1)} & "
            f"{cell(agg['boundary_error_m_mean'], 'boundary_error_m_mean', 1, 1)} & "
            f"{cell(agg['boundary_error_m_p95'], 'boundary_error_m_p95', 1, 1)} & "
            f"{cell(agg['pixel_iou'], 'pixel_iou', 1, 100)} & "
            f"{cell(agg['pq_small'], 'pq_small', 1, 100)} & "
            f"{cell(agg['pq_medium'], 'pq_medium', 1, 100)} & "
            f"{cell(agg['pq_large'], 'pq_large', 1, 100)} \\\\"
        )

    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l l ccc c c cc c ccc@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Panoptic} & & & "
        r"\multicolumn{2}{c}{\makecell{Bd.\ err\ (m)}} & & "
        r"\multicolumn{3}{c}{\makecell{PQ by GT field-area bin}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){8-9} \cmidrule(lr){11-13}",
        r"Method & Bb. & PQ & SQ & RQ$_{.5}$ & "
        r"F1$_{[.5{:}.95]}$ & \makecell{$|\Delta N|/N$} & mean & p95 & "
        r"\makecell{Pixel\\IoU$^{\dagger}$} & PQ$_\mathrm{s}$ & PQ$_\mathrm{m}$ & PQ$_\mathrm{l}$ \\",
        r"\midrule",
    ]
    for (name, bb, _poly, _a, _p, _b, bold_bb, sep), agg in zip(ROWS, aggs):
        if sep:
            lines.append(r"\midrule")
        lines.append(row_line(name, bb, bold_bb, agg))
    lines += [r"\bottomrule", r"\end{tabular}"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for (name, bb, *_), agg in zip(ROWS, aggs):
        plain = name.replace(r"\textbf{", "").replace("}", "").split("~")[0].replace("$^{*}$", "")
        print(
            f"  {plain:30s} {bb:7s} PQ={agg['pq'] * 100:5.1f} "
            f"pixIoU={agg['pixel_iou'] * 100:5.1f} |dN|/N={agg['dN_norm']:.2f} "
            f"PQ[s/m/l]={agg['pq_small'] * 100:.1f}/{agg['pq_medium'] * 100:.1f}/{agg['pq_large'] * 100:.1f}"
        )


if __name__ == "__main__":
    main()
