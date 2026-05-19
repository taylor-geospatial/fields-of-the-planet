"""Smallholder scatter: per-country median field size vs Delta Obj F1.

x-axis: per-country median FTW polygon area (hectares), computed by
reprojecting WGS84 polygons in ``data/ftw_polygons/<country>.parquet`` to
EPSG:6933 (NSIDC EASE-Grid 2.0 global equal-area) and taking the median
of the planar area in m^2 (then /1e4 to ha). We compute area from
geometry rather than the parquet ``area`` column because the latter is
missing for several countries and its units are inconsistent across
files.

y-axis: Delta Obj F1 = (Planet B3 augmax full WS+TTA) - (S2 PRUE-B7 full),
both on the FTW v3.1 ``full_data`` 22-country test split. Same sources
used by ``make_per_country_bars.py``:
- Planet: ``logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv``
- S2 PRUE-B7 full: ``logs/ftw_official/b7_<country>.csv``

Smallholder classification: a country is labeled "smallholder-dominated"
if its median FTW field area is < 2.0 ha. This 2 ha threshold follows
the Lowder et al. 2016 (FAO) operational definition of a smallholder
farm (\\cite{lowder2016farms} in paper/refs.bib).

Writes:
- paper/figs/smallholder_scatter.pdf
- paper/scripts/output/smallholder_scatter.csv
"""

import glob
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parents[1] / "figs"
OUT = Path(__file__).parent / "output"
FIGS.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

SMALLHOLDER_HA_THRESHOLD = 2.0  # Lowder et al. 2016 operational threshold

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def compute_median_field_ha() -> pd.DataFrame:
    """Median FTW polygon area (ha) per country, computed in EPSG:6933."""
    rows: list[dict[str, float | int | str]] = []
    for f in sorted((ROOT / "data" / "ftw_polygons").glob("*.parquet")):
        country = f.stem
        g = gpd.read_parquet(f)
        if g.crs is None:
            raise ValueError(f"{f} has no CRS")
        g2 = g.to_crs(6933)
        a_ha = g2.geometry.area / 1e4
        rows.append(
            {
                "country": country,
                "n_polygons": len(g),
                "median_field_size_ha": float(a_ha.median()),
            }
        )
    return pd.DataFrame(rows)


def load_s2_b7_full() -> pd.DataFrame:
    """S2 PRUE-B7 full per-country results (FTW v3.1 full_data)."""
    files = [
        f
        for f in sorted(glob.glob(str(ROOT / "logs/ftw_official/b7_*.csv")))
        if "per_country" not in f
    ]
    s2 = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s2 = s2.rename(columns={"countries": "country", "object_level_f1": "f1_s2"})
    s2 = s2.drop_duplicates(subset="country", keep="last")
    return s2[["country", "f1_s2"]]


def load_planet_full() -> pd.DataFrame:
    pl = pd.read_csv(ROOT / "logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv")
    return pl.rename(columns={"object_ws_f1": "f1_pl"})[["country", "f1_pl"]]


def build_table() -> pd.DataFrame:
    sizes = compute_median_field_ha()
    s2 = load_s2_b7_full()
    pl = load_planet_full()
    m = sizes.merge(s2, on="country", how="inner").merge(pl, on="country", how="inner")
    m["delta_obj_f1"] = m["f1_pl"] - m["f1_s2"]
    m["is_smallholder"] = m["median_field_size_ha"] < SMALLHOLDER_HA_THRESHOLD
    return m.sort_values("median_field_size_ha").reset_index(drop=True)


def _label_offsets(df: pd.DataFrame) -> dict[str, tuple[float, float, str, str]]:
    """Per-country hand offsets in axes-data units (dx, dy, ha, va).

    Tuned to avoid overlap on the final scatter. dx is log-multiplicative
    (added on log10(x)), dy is additive on y. Defaults: (+0.04, +0.012).
    """
    default = (0.04, 0.012, "left", "bottom")
    overrides: dict[str, tuple[float, float, str, str]] = {
        "cambodia": (-0.04, 0.010, "right", "bottom"),
        "vietnam": (0.04, -0.014, "left", "top"),
        "india": (-0.04, -0.014, "right", "top"),
        "spain": (0.04, -0.014, "left", "top"),
        "kenya": (0.04, -0.014, "left", "top"),
        "slovenia": (-0.04, -0.014, "right", "top"),
        "portugal": (0.04, -0.014, "left", "top"),
        "rwanda": (-0.04, 0.010, "right", "bottom"),
        "croatia": (0.04, 0.012, "left", "bottom"),
        "austria": (-0.04, 0.012, "right", "bottom"),
        "belgium": (0.04, -0.014, "left", "top"),
        "luxembourg": (-0.04, -0.014, "right", "top"),
        "lithuania": (0.04, 0.012, "left", "bottom"),
        "corsica": (-0.04, 0.012, "right", "bottom"),
        "netherlands": (0.04, -0.014, "left", "top"),
        "latvia": (0.04, 0.012, "left", "bottom"),
        "finland": (-0.04, 0.012, "right", "bottom"),
        "sweden": (0.04, 0.012, "left", "bottom"),
        "estonia": (-0.04, -0.014, "right", "top"),
        "france": (0.04, -0.014, "left", "top"),
        "denmark": (-0.04, 0.012, "right", "bottom"),
        "slovakia": (0.04, 0.012, "left", "bottom"),
        "south_africa": (-0.04, 0.012, "right", "bottom"),
        "germany": (0.04, -0.014, "left", "top"),
    }
    return {c: overrides.get(c, default) for c in df.country}


def make_plot(df: pd.DataFrame) -> None:
    color_sh = "#8b3a1f"  # sienna -- smallholder (median < 2 ha)
    color_other = "#5b7026"  # olive -- larger fields

    x = df["median_field_size_ha"].to_numpy()
    y = df["delta_obj_f1"].to_numpy()
    log_x = np.log10(x)

    # OLS fit in log10(ha) space (field size is heavy-tailed).
    slope, intercept, r_value, p_value, _ = stats.linregress(log_x, y)
    rho, rho_p = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # Trend line
    xfit = np.linspace(log_x.min() - 0.05, log_x.max() + 0.05, 100)
    yfit = slope * xfit + intercept
    ax.plot(
        10**xfit,
        yfit,
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.6,
        zorder=1,
    )

    for is_sh, color, lbl in [
        (True, color_sh, f"smallholder (median < {SMALLHOLDER_HA_THRESHOLD:g} ha)"),
        (False, color_other, f"larger fields (median ≥ {SMALLHOLDER_HA_THRESHOLD:g} ha)"),
    ]:
        sub = df[df.is_smallholder == is_sh]
        ax.scatter(
            sub.median_field_size_ha,
            sub.delta_obj_f1,
            s=46,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            label=lbl,
        )

    # Country labels
    offsets = _label_offsets(df)
    for _, row in df.iterrows():
        dx, dy, ha, va = offsets[row.country]
        ax.annotate(
            row.country.replace("_", " "),
            xy=(row.median_field_size_ha, row.delta_obj_f1),
            xytext=(10 ** (np.log10(row.median_field_size_ha) + dx), row.delta_obj_f1 + dy),
            ha=ha,
            va=va,
            fontsize=7,
            color="black",
            zorder=4,
        )

    ax.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax.axvline(
        SMALLHOLDER_HA_THRESHOLD,
        color="gray",
        linewidth=0.6,
        linestyle=":",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Median FTW field area (ha, log scale)")
    ax.set_ylabel(r"$\Delta$ Obj F1 (Planet B3 augmax full $-$ S2 PRUE-B7 full)")
    ax.grid(which="both", linewidth=0.3, alpha=0.5)

    # Stats box
    stat_txt = (
        f"OLS on $\\log_{{10}}$(ha): slope $=$ {slope:+.3f}, "
        f"$r$ $=$ {r_value:+.2f} ($p$ $=$ {p_value:.2g})\n"
        f"Spearman $\\rho$ $=$ {rho:+.2f} ($p$ $=$ {rho_p:.2g}); "
        f"$n$ $=$ {len(df)} countries"
    )
    ax.text(
        0.98,
        0.02,
        stat_txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.7",
            "linewidth": 0.5,
        },
    )

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.4)
    out_pdf = FIGS / "smallholder_scatter.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(
        f"OLS log10(ha): slope={slope:+.3f} intercept={intercept:+.3f} "
        f"r={r_value:+.3f} p={p_value:.3g}; spearman rho={rho:+.3f} p={rho_p:.3g}"
    )


def main() -> None:
    df = build_table()
    out_csv = OUT / "smallholder_scatter.csv"
    df[
        [
            "country",
            "n_polygons",
            "median_field_size_ha",
            "f1_pl",
            "f1_s2",
            "delta_obj_f1",
            "is_smallholder",
        ]
    ].to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(
        df[["country", "median_field_size_ha", "delta_obj_f1", "is_smallholder"]].to_string(
            index=False
        )
    )
    make_plot(df)


if __name__ == "__main__":
    main()
