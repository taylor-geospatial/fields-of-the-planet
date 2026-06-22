"""Generate ``paper/figs/polygon_metrics.tex`` (``tab:polygon_metrics``).

Dense-label held-out macro-average (HELDOUT_10_DENSE: the 11 held-out
countries minus kenya) of panoptic-quality, F1, normalized polygon-count
error, and meter-scale boundary error for each (imagery, recipe)
configuration, plus PQ broken down by GT field-area bin (small/medium/large).
Sources: ``logs/polygon_metrics/<stem>.csv`` and ``logs/area_bins/<stem>.csv.bins.csv``.

Kenya's labels are presence-only (background untrusted), so its supervised
polygon metrics are not comparable and it is excluded from the macro; see
the per-country table for the presence-only stress-test row. Boundary-error
entries for kenya are NaN in the source CSVs regardless.

Run::

    uv run python paper/scripts/polygon_metrics_table.py
"""

from pathlib import Path

import pandas as pd
from _aggregate import HELDOUT_10_DENSE, load_and_filter

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "paper" / "figs" / "polygon_metrics.tex"
SRC = REPO / "logs" / "polygon_metrics"
AREA = REPO / "logs" / "area_bins"

# The B3-full row is the released checkpoint (retrained Jun 2026, epoch 92);
# its metrics come from the reproduction eval rather than the original run.
REPRO = REPO / "logs" / "repro_eval"

# (model, backbone, csv path, midrule-before, bold-row, boundary-csv, area-stem).
# Full-data split only. S2 rows: PQ/SQ/RQ/F1/|dN| come from the upsample-512
# eval (resize_factor=2, how the PRUE checkpoints are run), but boundary error
# is taken from the NATIVE-grid eval -- the meter chamfer is grid-sensitive and
# the finer upsample grid inflates it (see app:upsampled_s2), so reporting it at
# each model's native grid is the like-for-like comparison. boundary-csv=None
# means use the row's own CSV for boundary too. area-stem=None means the model
# has no area-binned eval (DelineateAnything).
ROWS = [
    ("DelineateAnything$^{*}$", "YOLO11", SRC / "delineate_anything_conf0005.csv",
     False, False, None, None),
    ("FTW-PRUE+", "B3", SRC / "s2_b3_augmax_full_upsampled_22.csv", True, False,
     SRC / "s2_b3_augmax_full_native256.csv", "s2_b3"),
    ("FTW-PRUE+", "B7", SRC / "s2_upsampled_b7_augmax_full_22.csv", False, False,
     SRC / "s2_b7_augmax_full_native256.csv", "s2_b7"),
    ("FTP-PRUE+", "B3", REPRO / "polygon_metrics.csv", False, True, None, "planet_b3"),
]

COLS = (
    "pq",
    "pq_sq",
    "pq_rq",
    "ap_5_95",
    "polygon_count_delta_norm",
    "boundary_error_m_mean",
    "boundary_error_m_p95",
)
AREA_BINS = ("small", "medium", "large")


def _area_pq(stem: str) -> dict[str, float]:
    """Micro-pooled PQ per field-area bin for one model (or NaNs if absent)."""
    if stem is None:
        return {b: float("nan") for b in AREA_BINS}
    d = pd.read_csv(AREA / f"{stem}.csv.bins.csv").set_index("bin")
    return {b: float(d.loc[b, "pq"]) for b in AREA_BINS}


def main() -> None:
    bnd_cols = ("boundary_error_m_mean", "boundary_error_m_p95")
    aggregates: list[dict[str, float]] = []
    for _, _, csv_path, _, _, bnd_csv, area_stem in ROWS:
        sub = load_and_filter(csv_path, HELDOUT_10_DENSE)
        if len(sub) != len(HELDOUT_10_DENSE):
            raise RuntimeError(
                f"{csv_path}: macro over {len(sub)}/{len(HELDOUT_10_DENSE)} countries"
            )
        agg = {
            c: float(sub[c].mean(skipna=True))
            for c in COLS
            if c != "polygon_count_delta_norm"
        }
        # Normalized polygon-count error: per-country mean |dN| divided by the
        # per-country mean GT count, then macro-averaged. |dN| alone scales with
        # parcel density (Cambodia's ~600 parcels/patch dominate the raw mean),
        # so the normalized form is the density-invariant over/under-seg metric.
        agg["polygon_count_delta_norm"] = float(
            (sub["polygon_count_delta_mean"] / sub["n_gt_mean"]).mean()
        )
        bsub = sub
        if bnd_csv is not None:
            bsub = load_and_filter(bnd_csv, HELDOUT_10_DENSE)
            if len(bsub) != len(HELDOUT_10_DENSE):
                raise RuntimeError(f"{bnd_csv}: boundary macro over {len(bsub)} countries")
            for c in bnd_cols:
                agg[c] = float(bsub[c].mean(skipna=True))
        agg["_bnd_n"] = int(bsub["boundary_error_m_mean"].notna().sum())
        # PQ by GT field-area bin (micro-pooled over fields; see tab:area_bins).
        for b, v in _area_pq(area_stem).items():
            agg[f"pq_{b}"] = v
        aggregates.append(agg)

    # Best per column (higher-is-better for PQ/SQ/RQ/F1 and the area-PQ bins;
    # lower-is-better for |dN|/N and boundary errors).
    area_cols = tuple(f"pq_{b}" for b in AREA_BINS)
    higher_better = {"pq", "pq_sq", "pq_rq", "ap_5_95", *area_cols}
    all_cols = (*COLS, *area_cols)
    best: dict[str, float] = {}
    for c in all_cols:
        vals = [a[c] for a in aggregates if a[c] == a[c]]  # skip NaN
        best[c] = (max(vals) if c in higher_better else min(vals)) if vals else float("nan")

    # [0,1] metrics (PQ/SQ/RQ/F1, area PQ) are shown x100 at 1 decimal; the
    # normalized count error and meter-scale boundary errors keep natural units.
    def cell(v: float, c: str, decimals: int = 1, scale: float = 100.0) -> str:
        if v != v:  # NaN -> no area eval for this model
            return "--"
        s = f"{v * scale:.{decimals}f}"
        if abs(v - best[c]) < 1e-9:
            s = rf"\textbf{{{s}}}"
        return s

    def row_line(model: str, backbone: str, bold: bool, agg: dict[str, float]) -> str:
        m, b = (
            (rf"\textbf{{{x}}}" for x in (model, backbone)) if bold else (model, backbone)
        )
        return (
            f"{m} & {b} & "
            f"{cell(agg['pq'], 'pq')} & {cell(agg['pq_sq'], 'pq_sq')} & "
            f"{cell(agg['pq_rq'], 'pq_rq')} & {cell(agg['ap_5_95'], 'ap_5_95')} & "
            f"{cell(agg['polygon_count_delta_norm'], 'polygon_count_delta_norm', 2, 1.0)} & "
            f"{cell(agg['boundary_error_m_mean'], 'boundary_error_m_mean', 1, 1.0)} & "
            f"{cell(agg['boundary_error_m_p95'], 'boundary_error_m_p95', 1, 1.0)} & "
            f"{cell(agg['pq_small'], 'pq_small')} & {cell(agg['pq_medium'], 'pq_medium')} & "
            f"{cell(agg['pq_large'], 'pq_large')} \\\\"
        )

    lines: list[str] = []
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{@{}l l ccc c c cc ccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{Panoptic} & & & "
        r"\multicolumn{2}{c}{\makecell{Bd.\ err\ (m)}} & "
        r"\multicolumn{3}{c}{\makecell{PQ by GT area}} \\"
    )
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){8-9} \cmidrule(lr){10-12}")
    lines.append(
        r"Model & Backbone & PQ & SQ & \makecell{RQ\\($=$F1$_{.5}$)} & "
        r"F1$_{[.5{:}.95]}$ & \makecell{$|\Delta N|/N$} & mean & p95 & "
        r"Sm. & Med. & Lg. \\"
    )
    lines.append(r"\midrule")
    for (model, backbone, _, sep, bold, _, _), agg in zip(ROWS, aggregates):
        if sep:
            lines.append(r"\midrule")
        lines.append(row_line(model, backbone, bold, agg))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    for (model, backbone, _, _, _, _, _), agg in zip(ROWS, aggregates):
        print(
            f"  {model} {backbone}: PQ={agg['pq']:.3f} "
            f"|dN|/N={agg['polygon_count_delta_norm']:.3f} "
            f"PQ[s/m/l]={agg['pq_small']:.3f}/{agg['pq_medium']:.3f}/{agg['pq_large']:.3f}"
        )


if __name__ == "__main__":
    main()
