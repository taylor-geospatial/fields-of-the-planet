"""Coverage map figures: FTW patch footprints over an imagery basemap.

Produces two outputs:

* ``coverage_single.pdf``  - one large country (default: Austria) shown as
  a translucent polygon swarm of every patch footprint over an Esri
  World Imagery basemap. Train/val/test splits colored independently.
* ``coverage_grid.pdf``    - 6-country grid showing geographic diversity.

Both use contextily + a polished theme; train/val/test split colors come
from the Okabe-Ito palette so the figures share a visual language with
the rest of the paper.
"""

import argparse
from pathlib import Path

import contextily as ctx  # contextily not in main CI deps; paper-scripts only
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

# Okabe-Ito-ish palette; train = olive (FTW-HD theme), test = sienna,
# val = a muted teal so the three splits never blend visually.
SPLIT_COLORS = {
    "train": "#5b7026",
    "val": "#117777",
    "test": "#b04a1c",
}


def _load_country(ftw_root: Path, country: str) -> gpd.GeoDataFrame:
    p = ftw_root / country / f"chips_{country}.parquet"
    g = gpd.read_parquet(p)
    return g.to_crs(3857)  # web-mercator to match the tile provider


def _plot_country(
    ax,
    gdf: gpd.GeoDataFrame,
    title: str,
    *,
    alpha: float = 0.45,
    edge_lw: float = 0.15,
    basemap_zoom: int | None = None,
) -> None:
    for split, color in SPLIT_COLORS.items():
        sub = gdf[gdf["split"] == split]
        if sub.empty:
            continue
        sub.plot(
            ax=ax,
            facecolor=color,
            edgecolor="white",
            linewidth=edge_lw,
            alpha=alpha,
        )
    # Esri World Imagery for the basemap. CartoDB.Voyager looks more
    # cartographic but the satellite imagery underneath is the right
    # mood for a remote-sensing paper.
    if basemap_zoom is None:
        # contextily's auto-zoom returns nonsense for small-extent country
        # polygons in EPSG:3857 (~10^6 meter bounds). Pick a sensible zoom
        # from the bbox extent in meters; tested across Rwanda (small) to
        # Brazil (continental).
        xmin, ymin, xmax, ymax = ax.axis()
        extent_m = max(xmax - xmin, ymax - ymin)
        if extent_m < 2.5e5:  # < 250 km
            basemap_zoom = 9
        elif extent_m < 1.5e6:  # < 1500 km
            basemap_zoom = 7
        else:
            basemap_zoom = 5
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,  # ty: ignore[unresolved-attribute]  # xyzservices stubs incomplete
            zoom=basemap_zoom,
        )
    except Exception as err:
        print(f"  basemap unavailable for {title}: {err}")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def _shared_legend(fig, ncol: int = 3) -> None:
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.4, alpha=0.6, label=s)
        for s, c in SPLIT_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )


def make_single(ftw_root: Path, country: str, out: Path) -> None:
    g = _load_country(ftw_root, country)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    _plot_country(ax, g, f"FTW-HD coverage: {country.replace('_', ' ').title()} (n={len(g)})")
    _shared_legend(fig, ncol=3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def make_grid(ftw_root: Path, countries: list[str], out: Path, n_cols: int = 3) -> None:
    n = len(countries)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.4))
    axes = axes.flatten()
    for ax, country in zip(axes, countries):
        g = _load_country(ftw_root, country)
        _plot_country(
            ax,
            g,
            f"{country.replace('_', ' ').title()} (n={len(g)})",
            alpha=0.55,
            edge_lw=0.1,
        )
    for ax in axes[len(countries) :]:
        ax.axis("off")
    _shared_legend(fig, ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.4)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ftw-root", type=Path, default=Path("data/ftw"))
    p.add_argument("--single-country", default="austria")
    p.add_argument(
        "--grid-countries",
        nargs="+",
        default=["austria", "france", "cambodia", "rwanda", "brazil", "finland"],
    )
    p.add_argument("--out-single", type=Path, default=Path("paper/figs/coverage_single.pdf"))
    p.add_argument("--out-grid", type=Path, default=Path("paper/figs/coverage_grid.pdf"))
    args = p.parse_args()
    make_single(args.ftw_root, args.single_country, args.out_single)
    make_grid(args.ftw_root, args.grid_countries, args.out_grid)


if __name__ == "__main__":
    main()
