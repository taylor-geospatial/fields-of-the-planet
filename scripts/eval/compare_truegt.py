"""Compare paper polygon-metric evals scored against the rasterized GT mask vs
the true FTW vector polygons. Reads whatever CSV pairs exist and writes a
Markdown report. Defensive: missing pairs are skipped and noted.

Pairs (rasterized -> true):
  Headline 23-region:
    planet  logs/polygon_metrics/planet_b3_rastergt_22.csv      -> planet_b3_truegt_22.csv
    s2-b7   logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv -> s2_upsampled_b7_truegt_22.csv
  Pooled area bins (dense-10):
    {planet_b3,s2_b7,s2_b3}.csv.bins.csv -> *_truegt.csv.bins.csv  (logs/area_bins)
  Per-country small-field:
    {planet_b3,s2_b7}_full23_small.csv -> *_full23_small_truegt.csv (logs/area_bins_per_country)
"""

from pathlib import Path

import pandas as pd

LOGS = Path("logs")
OUT = LOGS / "truegt_comparison.md"


def _macro(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if col in df and len(df) else float("nan")


def _headline(lines: list[str], name: str, ras_p: Path, true_p: Path) -> None:
    if not (ras_p.exists() and true_p.exists()):
        lines.append(
            f"### {name}\n\n_skipped_ (missing: "
            f"{'ras ' if not ras_p.exists() else ''}{'true' if not true_p.exists() else ''})\n"
        )
        return
    r = pd.read_csv(ras_p)
    t = pd.read_csv(true_p)
    cols = ["country", "pq", "ap_5_95", "boundary_error_m_mean", "n_gt_mean"]
    m = r[cols].merge(t[cols], on="country", suffixes=("_ras", "_true"))
    lines.append(f"### {name}  (n={len(m)} regions)\n")
    lines.append("| metric | rasterized | true-poly | Δ (true-ras) |")
    lines.append("|---|---|---|---|")
    for label, col in [("PQ", "pq"), ("AP", "ap_5_95")]:
        ras, tru = _macro(m, f"{col}_ras") * 100, _macro(m, f"{col}_true") * 100
        lines.append(f"| macro {label} | {ras:.1f} | {tru:.1f} | {tru - ras:+.1f} |")
    br, bt = _macro(m, "boundary_error_m_mean_ras"), _macro(m, "boundary_error_m_mean_true")
    lines.append(f"| boundary err (m) | {br:.2f} | {bt:.2f} | {bt - br:+.2f} |")
    nr, nt = _macro(m, "n_gt_mean_ras"), _macro(m, "n_gt_mean_true")
    lines.append(f"| mean #GT / patch | {nr:.1f} | {nt:.1f} | {nt - nr:+.1f} |")
    # per-country PQ delta, sorted by magnitude
    m["dPQ"] = (m.pq_true - m.pq_ras) * 100
    big = m.reindex(m.dPQ.abs().sort_values(ascending=False).index).head(8)
    lines.append(
        "\nLargest per-region PQ shifts (true-ras, pp): "
        + ", ".join(f"{x.country} {x.dPQ:+.1f}" for _, x in big.iterrows())
        + "\n"
    )


def _bins(lines: list[str], name: str, ras_p: Path, true_p: Path) -> None:
    if not (ras_p.exists() and true_p.exists()):
        lines.append(f"### {name} area bins\n\n_skipped_ (missing files)\n")
        return
    r = pd.read_csv(ras_p).set_index("bin")
    t = pd.read_csv(true_p).set_index("bin")
    lines.append(f"### {name} — pooled PQ by field-area bin\n")
    lines.append("| bin | PQ ras | PQ true | Δ | AP ras | AP true | Δ | n_gt ras | n_gt true |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in ["small", "medium", "large", "all"]:
        if b not in r.index or b not in t.index:
            continue
        rr, tt = r.loc[b], t.loc[b]
        lines.append(
            f"| {b} | {rr.pq * 100:.1f} | {tt.pq * 100:.1f} | {(tt.pq - rr.pq) * 100:+.1f} "
            f"| {rr.ap_5_95 * 100:.1f} | {tt.ap_5_95 * 100:.1f} | {(tt.ap_5_95 - rr.ap_5_95) * 100:+.1f} "
            f"| {int(rr.n_gt)} | {int(tt.n_gt)} |"
        )
    lines.append("")


def _percountry(lines: list[str], name: str, ras_p: Path, true_p: Path) -> None:
    if not (ras_p.exists() and true_p.exists()):
        lines.append(f"### {name} per-country small-field\n\n_skipped_ (missing files)\n")
        return
    r = pd.read_csv(ras_p)
    t = pd.read_csv(true_p)
    m = r.merge(t, on="country", suffixes=("_ras", "_true"))
    lines.append(f"### {name} — per-country small-field PQ (n={len(m)})\n")
    rmac, tmac = m.pq_small_ras.mean() * 100, m.pq_small_true.mean() * 100
    lines.append(
        f"macro small-field PQ: rasterized {rmac:.1f} -> true {tmac:.1f} ({tmac - rmac:+.1f} pp)\n"
    )


def main() -> int:
    pm = LOGS / "polygon_metrics"
    ab = LOGS / "area_bins"
    pc = LOGS / "area_bins_per_country"
    lines = [
        "# True-polygon vs rasterized-mask GT — polygon metric comparison\n",
        "Same model + flags; only the GT source differs.\n",
    ]

    lines.append("## Headline (23 regions)\n")
    _headline(
        lines, "PlanetScope B3", pm / "planet_b3_rastergt_22.csv", pm / "planet_b3_truegt_22.csv"
    )
    _headline(
        lines,
        "Sentinel-2 B7 (upsampled)",
        pm / "s2_upsampled_b7_augmax_full_22.csv",
        pm / "s2_upsampled_b7_truegt_22.csv",
    )

    lines.append("\n## Area bins (dense-10, pooled)\n")
    _bins(
        lines, "PlanetScope B3", ab / "planet_b3.csv.bins.csv", ab / "planet_b3_truegt.csv.bins.csv"
    )
    _bins(lines, "Sentinel-2 B7", ab / "s2_b7.csv.bins.csv", ab / "s2_b7_truegt.csv.bins.csv")
    _bins(lines, "Sentinel-2 B3", ab / "s2_b3.csv.bins.csv", ab / "s2_b3_truegt.csv.bins.csv")

    lines.append("\n## Per-country small-field\n")
    _percountry(
        lines,
        "PlanetScope B3",
        pc / "planet_b3_full23_small.csv",
        pc / "planet_b3_full23_small_truegt.csv",
    )
    _percountry(
        lines, "Sentinel-2 B7", pc / "s2_b7_full23_small.csv", pc / "s2_b7_full23_small_truegt.csv"
    )

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
