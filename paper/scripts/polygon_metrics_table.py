"""Generate ``paper/figs/polygon_metrics.tex`` (``tab:polygon_metrics``).

Dense-label held-out macro-average (HELDOUT_10_DENSE) of polygon metrics, the
secondary pixel IoU, and PQ by GT field-area bin, for each method:

* DelineateAnything / DelineateAnything-S (YOLO11x / YOLO11n), zero-shot, on
  PlanetScope; sensor is folded into the method name. These baselines retain
  their original rasterized-GT scores and are NOT re-scored against polygons.
* FTW-PRUE+ B3/B7 (Sentinel-2) and FTP-PRUE+ B3/B7 (PlanetScope, ours).

The four segmentation rows are scored against the TRUE FTW vector polygons at
each sensor's native ground resolution (Sentinel-2 capped to 10 m via
``--score-gsd-m 10``); those polygon metrics + PQ-by-area-bin come from the
per-country ``logs/resolution_ablation/<condition>/`` runs (macro-averaged).
Boundary chamfer (native grid) and pixel IoU stay on their established sources.

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
PP = REPO / "logs" / "postproc_ablation"
REPRO = REPO / "logs" / "repro_eval"
# Native-GSD true-GT per-country runs (one CSV + .bins.csv per dense-10 country).
RESABL = REPO / "logs" / "resolution_ablation"

_DA = r"~\cite{lavreniuk2025delineate}"
_PRUE = r"~\cite{muhawenayo2026prue}"

# Each row: display name, backbone, polygon-metrics source, area-bins source,
# pixel-IoU CSV (pixel_level_iou col), boundary CSV (None = use polygon source),
# bold-backbone, midrule-before. A polygon/area-bins source may be a single CSV
# (rasterized DA baselines) or a per-country DIRECTORY (native-GSD seg rows).
ROWS = [
    (
        rf"DelineateAnything (Planet)$^{{*}}${_DA}",
        "YOLO11x",
        PM / "delineate_x_planet.csv",
        None,
        PM / "delineate_x_planet.csv",
        None,
        False,
        False,
    ),
    (
        rf"DelineateAnything-S (Planet)$^{{*}}${_DA}",
        "YOLO11n",
        PM / "delineate_s_planet.csv",
        None,
        PM / "delineate_s_planet.csv",
        None,
        False,
        False,
    ),
    (
        rf"FTW-PRUE+ (S2){_PRUE}",
        "B3",
        RESABL / "s2b3_10m",
        RESABL / "s2b3_10m",
        PP / "s2_b3_augmax_full_upsampled_ws_tta.csv",
        PM / "s2_b3_augmax_full_native256.csv",
        False,
        True,
    ),
    (
        rf"FTW-PRUE+ (S2){_PRUE}",
        "B7",
        RESABL / "s2nat10",
        RESABL / "s2nat10",
        PP / "s2_b7_augmax_full_upsampled_ws_tta.csv",
        PM / "s2_b7_augmax_full_native256.csv",
        False,
        False,
    ),
    (
        r"\textbf{FTP-PRUE+ (Planet, ours)}",
        "B3",
        RESABL / "planet3m",
        RESABL / "planet3m",
        REPRO / "pp_ws_tta.csv",
        REPRO / "polygon_metrics.csv",
        True,
        True,
    ),
    (
        r"\textbf{FTP-PRUE+ (Planet, ours)}",
        "B7",
        RESABL / "planetb7_3m",
        RESABL / "planetb7_3m",
        PP / "planet_b7_augmax_full_ws_tta.csv",
        REPRO / "polygon_metrics.csv",
        True,
        False,
    ),
]

POLY_COLS = ("pq", "pq_sq", "pq_rq", "ap_5_95")
BND_COLS = ("boundary_error_m_mean", "boundary_error_m_p95")
AREA_BINS = ("small", "medium", "large")


def _country_csvs(d: Path) -> list[Path]:
    files = sorted(p for p in d.glob("*.csv") if not p.name.endswith(".bins.csv"))
    if len(files) != len(HELDOUT_10_DENSE):
        raise RuntimeError(f"{d}: {len(files)} per-country CSVs, expected {len(HELDOUT_10_DENSE)}")
    return files


def _macro(src: Path, col: str) -> float:
    if src.is_dir():
        vals = [float(pd.read_csv(c)[col].iloc[0]) for c in _country_csvs(src)]
        return sum(vals) / len(vals)
    sub = load_and_filter(src, HELDOUT_10_DENSE)
    if len(sub) != len(HELDOUT_10_DENSE):
        raise RuntimeError(f"{src}: macro over {len(sub)}/{len(HELDOUT_10_DENSE)} countries")
    return float(sub[col].mean(skipna=True))


def _norm_count(src: Path) -> float:
    if src.is_dir():
        vals = []
        for c in _country_csvs(src):
            df = pd.read_csv(c)
            vals.append(
                abs(df["n_pred_mean"].iloc[0] - df["n_gt_mean"].iloc[0]) / df["n_gt_mean"].iloc[0]
            )
        return sum(vals) / len(vals)
    sub = load_and_filter(src, HELDOUT_10_DENSE)
    return float((sub["polygon_count_delta_mean"] / sub["n_gt_mean"]).mean())


def _area_pq(src: Path) -> dict[str, float]:
    if src.is_dir():
        out: dict[str, float] = {}
        for b in AREA_BINS:
            vals = []
            for c in _country_csvs(src):
                d = pd.read_csv(Path(f"{c}.bins.csv")).set_index("bin")
                vals.append(float(d.loc[b, "pq"]))
            out[b] = sum(vals) / len(vals)
        return out
    d = pd.read_csv(src).set_index("bin")
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
        if area_csv is None:  # baselines not re-scored against polygons
            for b in AREA_BINS:
                agg[f"pq_{b}"] = float("nan")
        else:
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
        r"\multicolumn{3}{c}{\makecell{PQ by GT size$^{\ddagger}$}} \\",
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
