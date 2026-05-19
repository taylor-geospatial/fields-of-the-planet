"""Dataset missingness / selection-bias audit.

Reconciles raw FTW Planet targets against the final usable index to
answer: are dropped patches systematically biased toward harder
countries, cloudier scenes, or denser-label regions?

Inputs (all under data/planet/_global/, plus data/planet/index.parquet):
  manifest.jsonl           per-(patch, window) target list
  completion_log.jsonl     extraction failures (status=exhausted)
  udm2_quality.jsonl       per-(patch, window) UDM2 usable_flag
  resample_log.jsonl       recovery attempts
  rasterize_summary.jsonl  per-tile field_pixels / total_pixels
  ../index.parquet         final index w/ usable_pair, cloud_cover_*

Outputs (paper/scripts/output/missingness_audit/):
  per_country.csv      counts + percentages per country
  cloud_density.csv    cross-tab: cloud-bin x kept/dropped, density-bin x kept/dropped
  stacked_bars.pdf     per-country stacked failure-mode bars

Prints a short text summary to stdout (worst 3 retention countries,
mean cloud + mean label density for kept vs dropped, MWU p-values).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ROOT = Path(__file__).resolve().parents[2]
GLOBAL = ROOT / "data" / "planet" / "_global"
INDEX = ROOT / "data" / "planet" / "index.parquet"
OUT = Path(__file__).parent / "output" / "missingness_audit"
OUT.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path, *, skip_bad: bool = False):
    """Yield JSON records from a JSONL file. Optionally skip malformed lines."""
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if skip_bad:
                    print(f"  warn: skipping malformed line {i} in {path.name}")
                    continue
                raise


def load_manifest_targets() -> pd.DataFrame:
    rows = [(r["country"], r["id"], r["window"]) for r in iter_jsonl(GLOBAL / "manifest.jsonl")]
    return pd.DataFrame(rows, columns=["country", "id", "window"])


def load_extraction_failed() -> set:
    out = set()
    for r in iter_jsonl(GLOBAL / "completion_log.jsonl"):
        if r["status"] == "exhausted":
            out.add((r["country"], r["id"], r["window"]))
    return out


def load_udm2_failed() -> set:
    """Return keys whose UDM2 quality says drop, OR whose UDM2 file failed to open."""
    out = set()
    for r in iter_jsonl(GLOBAL / "udm2_quality.jsonl"):
        if "usable_flag" in r:
            if not r["usable_flag"]:
                out.add((r["country"], r["id"], r["window"]))
            continue
        # Failed-to-open rows have only {path, status, error}. Derive key
        # from the path: data/planet/<country>/<id>_<window>_udm2.tif
        if r.get("status") != "failed":
            continue
        name = Path(r["path"]).name  # <id>_<window>_udm2.tif
        country = Path(r["path"]).parent.name
        stem = name[: -len("_udm2.tif")]
        # window is trailing _a or _b
        window = stem[-1]
        pid = stem[:-2]
        out.add((country, pid, window))
    return out


def load_resample_status() -> dict:
    """Map (country, id, window) -> last status string."""
    last = {}
    for r in iter_jsonl(GLOBAL / "resample_log.jsonl", skip_bad=True):
        key = (r["country"], r["id"], r["window"])
        last[key] = r["status"]
    return last


def load_label_density() -> dict:
    """Map patch_id -> field_pixel_fraction (from rasterize_summary).

    Same label tile is rasterized for both windows; one entry per patch
    suffices. Take the largest `field_pixels / total_pixels` we see.
    """
    out: dict[str, float] = {}
    for r in iter_jsonl(GLOBAL / "rasterize_summary.jsonl"):
        if r.get("status") != "ok":
            continue
        path = r["path"]
        # path: data/planet/<country>/<patch>_<a|b>_label.tif
        name = Path(path).name
        # strip _<a|b>_label.tif
        if name.endswith("_a_label.tif"):
            pid = name[: -len("_a_label.tif")]
        elif name.endswith("_b_label.tif"):
            pid = name[: -len("_b_label.tif")]
        else:
            continue
        total = r.get("total_pixels") or 0
        if total <= 0:
            continue
        frac = r["field_pixels"] / total
        prev = out.get(pid)
        if prev is None or frac > prev:
            out[pid] = frac
    return out


def main():
    print("Loading inputs ...")
    manifest = load_manifest_targets()
    print(f"  manifest rows: {len(manifest):,}")

    extraction_failed = load_extraction_failed()
    print(f"  extraction_failed: {len(extraction_failed):,}")

    udm2_failed = load_udm2_failed()
    print(f"  udm2_failed: {len(udm2_failed):,}")

    resample_last = load_resample_status()
    print(f"  resample log entries (unique keys): {len(resample_last):,}")

    label_density = load_label_density()
    print(f"  rasterize_summary patches w/ density: {len(label_density):,}")

    idx = pq.read_table(
        INDEX,
        columns=[
            "patch_id",
            "country",
            "cloud_cover_a",
            "cloud_cover_b",
            "usable_pair",
        ],
    ).to_pandas()
    idx["max_cloud"] = idx[["cloud_cover_a", "cloud_cover_b"]].max(axis=1)
    print(f"  index.parquet rows: {len(idx):,}  (usable_pair=True: {idx['usable_pair'].sum():,})")

    # Mark each (country, id, window) target with its outcome.
    # Priority: extraction_failed > udm2_failed (and not recovered) > kept.
    # "resample_exhausted" = udm2_failed AND resample_last status in {open_failed, no_url, no_candidate, extract_failed}.
    recovered = {"matched", "matched_new"}
    bad_recovery = {"open_failed", "no_url", "no_candidate", "extract_failed"}

    countries = manifest["country"].to_numpy()
    ids = manifest["id"].to_numpy()
    windows = manifest["window"].to_numpy()

    outcomes: list[str] = []
    for country, pid, window in zip(countries, ids, windows, strict=True):
        key = (country, pid, window)
        if key in extraction_failed:
            outcomes.append("extraction_failed")
        elif key in udm2_failed:
            st = resample_last.get(key)
            if st in recovered:
                outcomes.append("kept")
            elif st in bad_recovery:
                outcomes.append("resample_exhausted")
            else:
                # udm2 failed, no resample attempt logged -> still dropped
                outcomes.append("udm2_failed")
        else:
            outcomes.append("kept")
    manifest["outcome"] = outcomes

    # Cross-check: drop a patch if EITHER window is non-kept (usable_pair semantics).
    by_patch_outcome: dict[tuple, set] = defaultdict(set)
    for country, pid, outcome in zip(countries, ids, outcomes, strict=True):
        by_patch_outcome[(country, pid)].add(outcome)

    def patch_label(states: set) -> str:
        # Worst-of-pair label for per-patch counts.
        for s in ("extraction_failed", "resample_exhausted", "udm2_failed"):
            if s in states:
                return s
        return "kept"

    patch_rows = [(c, pid, patch_label(states)) for (c, pid), states in by_patch_outcome.items()]
    patch_df = pd.DataFrame(patch_rows, columns=["country", "patch_id", "outcome"])

    # Per-country breakdown (patch-level).
    per_country = (
        patch_df.groupby(["country", "outcome"]).size().unstack(fill_value=0).reset_index()
    )
    for col in ["extraction_failed", "udm2_failed", "resample_exhausted", "kept"]:
        if col not in per_country.columns:
            per_country[col] = 0
    per_country["total_targets"] = (
        per_country["extraction_failed"]
        + per_country["udm2_failed"]
        + per_country["resample_exhausted"]
        + per_country["kept"]
    )
    for col in ["extraction_failed", "udm2_failed", "resample_exhausted", "kept"]:
        per_country[f"{col}_pct"] = 100.0 * per_country[col] / per_country["total_targets"]
    per_country = per_country.sort_values("kept_pct").reset_index(drop=True)
    per_country.to_csv(OUT / "per_country.csv", index=False)
    print(f"\nWrote {OUT / 'per_country.csv'}")

    # Worst-3 retention.
    worst = per_country.head(3)[["country", "total_targets", "kept", "kept_pct"]]
    print("\nWorst-3 retention countries:")
    print(worst.to_string(index=False))

    # Build a per-patch table with cloud + label density + dropped/kept flag.
    # NB: patch_id is not globally unique — 3.4k ids span multiple countries.
    # Always key on (country, patch_id).
    idx_small = idx[["patch_id", "country", "max_cloud", "usable_pair"]].copy()
    idx_small["label_density"] = idx_small["patch_id"].map(label_density)
    kept_keys = set(
        map(tuple, idx_small.loc[idx_small["usable_pair"], ["country", "patch_id"]].values)
    )
    patch_df["dropped"] = ~patch_df.apply(
        lambda r: (r["country"], r["patch_id"]) in kept_keys, axis=1
    )

    # Attach cloud + density via (country, patch_id).
    merged = patch_df.merge(
        idx_small[["country", "patch_id", "max_cloud", "label_density"]],
        on=["country", "patch_id"],
        how="left",
    )

    obs = merged.dropna(subset=["max_cloud"]).copy()
    kept_obs = obs[~obs["dropped"]]
    drop_obs = obs[obs["dropped"]]

    print(
        f"\nPatches with cloud_cover observation: kept={len(kept_obs):,}, "
        f"dropped={len(drop_obs):,} (extraction-failed patches absent here)."
    )

    def summarize(name: str, kept_vals: pd.Series, drop_vals: pd.Series) -> dict:
        kept_vals = kept_vals.dropna()
        drop_vals = drop_vals.dropna()
        if len(kept_vals) == 0 or len(drop_vals) == 0:
            return {
                "metric": name,
                "kept_mean": float("nan"),
                "drop_mean": float("nan"),
                "p": float("nan"),
            }
        u, p = mannwhitneyu(drop_vals, kept_vals, alternative="two-sided")
        return {
            "metric": name,
            "kept_n": len(kept_vals),
            "drop_n": len(drop_vals),
            "kept_mean": float(kept_vals.mean()),
            "drop_mean": float(drop_vals.mean()),
            "u": float(u),
            "p": float(p),
        }

    stat_rows = [
        summarize("cloud_cover", kept_obs["max_cloud"], drop_obs["max_cloud"]),
        summarize("label_density", kept_obs["label_density"], drop_obs["label_density"]),
    ]
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(OUT / "kept_vs_dropped_stats.csv", index=False)
    print("\nKept vs dropped (Mann-Whitney U, two-sided):")
    print(stats_df.to_string(index=False))

    # Cross-tabs: cloud bin x dropped, density bin x dropped.
    # Search-side filter caps scene cloud cover at 0.10, so bins below that.
    cloud_bins = [-0.001, 0.01, 0.03, 0.05, 0.08, 0.101]
    dens_bins = [-0.001, 0.01, 0.05, 0.15, 0.4, 1.001]
    obs["cloud_bin"] = pd.cut(obs["max_cloud"], cloud_bins)
    obs["density_bin"] = pd.cut(obs["label_density"], dens_bins)

    def crosstab(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
        tab = (
            df.dropna(subset=[bin_col])
            .groupby([bin_col, "dropped"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        # Columns are bool {False, True}; map to named cols explicitly.
        kept_col = tab.get(False, pd.Series(0, index=tab.index))
        drop_col = tab.get(True, pd.Series(0, index=tab.index))
        out_tab = pd.DataFrame({"kept": kept_col, "dropped": drop_col})
        out_tab["drop_rate_pct"] = 100.0 * out_tab["dropped"] / out_tab.sum(axis=1)
        return out_tab

    cloud_tab = crosstab(obs, "cloud_bin")
    cloud_tab.to_csv(OUT / "cloud_crosstab.csv")
    dens_tab = crosstab(obs, "density_bin")
    dens_tab.to_csv(OUT / "density_crosstab.csv")

    print("\nCloud-cover bin -> drop rate:")
    print(cloud_tab.to_string())
    print("\nLabel-density bin -> drop rate:")
    print(dens_tab.to_string())

    # Stacked bars per country.
    plot_df = per_country.set_index("country")[
        ["kept", "udm2_failed", "resample_exhausted", "extraction_failed"]
    ]
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bottom = np.zeros(len(plot_df))
    x = np.arange(len(plot_df))
    for col, color in zip(plot_df.columns, colors, strict=True):
        ax.bar(x, plot_df[col].values, bottom=bottom, color=color, label=col, width=0.78)
        bottom = bottom + plot_df[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    ax.set_ylabel("patches")
    ax.set_title("Per-country target outcomes (sorted by retention rate)")
    ax.legend(loc="upper left", frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "stacked_bars.pdf")
    fig.savefig(OUT / "stacked_bars.png", dpi=180)
    plt.close(fig)
    print(f"\nWrote {OUT / 'stacked_bars.pdf'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
