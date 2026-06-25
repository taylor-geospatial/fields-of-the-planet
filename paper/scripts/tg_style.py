"""Shared Taylor Geospatial figure theme for the FTP paper.

One palette and one set of matplotlib defaults so every figure reads as a
coherent set. Charts pull series colors from ``SERIES``; image/label figures
share ``label_cmap()`` for the 3-class field mask.

Fonts are left to each figure so the paper charts keep their serif body face;
the palette is what carries the brand identity. No logos or wordmarks are added:
the paper is a double-blind submission.
"""

from matplotlib.colors import ListedColormap

# Taylor Geospatial palette (Brand Guidelines v4.3.26).
BROWN = "#3b1e1c"
IVORY = "#f4f4eb"
PERIWINKLE = "#80a0d8"
RED = "#ff4f2c"
LIGHT_BLUE = "#a7d0dc"
GREEN = "#cff29e"

# Darker variants for ink on the ivory page (the pale tints wash out as lines).
PERIWINKLE_INK = "#5a7ab8"
GREEN_INK = "#7cbf4f"

# Semantic series colors, shared across every chart so a method keeps its color.
# Sentinel-2 / 10 m baseline reads cool; Planet / our 3 m result reads in the
# brand highlight red so it pops as the contribution.
SERIES = {
    "s2": PERIWINKLE_INK,
    "planet": RED,
    "baseline": PERIWINKLE_INK,
    "ours": RED,
}

# Categorical cycle for charts with more than two series.
CYCLE = [RED, PERIWINKLE_INK, BROWN, LIGHT_BLUE, GREEN_INK]

# 3-class field label: background / interior / boundary.
LABEL_COLORS = [BROWN, GREEN, RED]


def label_cmap() -> ListedColormap:
    return ListedColormap(LABEL_COLORS)


def glasbey_colors(n: int):
    """First ``n`` Glasbey colors (cycled past 256) as an (n, 3) RGB float array.

    Glasbey is a large categorical palette of maximally perceptually-distinct
    hues -- used for per-instance field coloring so adjacent fields stay
    distinguishable without the repeats of a 10/20-color cycle. colorcet is
    imported lazily, so only figure generation needs it installed.
    """
    import colorcet as cc
    import numpy as np
    from matplotlib.colors import to_rgb

    pal = [to_rgb(c) for c in cc.glasbey]
    return np.array([pal[i % len(pal)] for i in range(max(1, n))], dtype="float32")


def instance_cmap(n: int, bg) -> ListedColormap:
    """ListedColormap with ``bg`` at index 0 (label 0 = background) followed by
    ``n`` Glasbey colors for instance labels 1..n."""
    import numpy as np

    return ListedColormap(np.vstack([np.asarray(bg, dtype="float32"), glasbey_colors(n)]))


def apply_style() -> None:
    """Set brand colors and clean axes. Font is left to the caller so paper
    figures keep the serif body face; only color and spine styling change."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=CYCLE),
            "axes.edgecolor": BROWN,
            "axes.labelcolor": BROWN,
            "axes.titlecolor": BROWN,
            "text.color": BROWN,
            "xtick.color": BROWN,
            "ytick.color": BROWN,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#d9d6c8",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )
