"""plotnine (Python ggplot2) versions of the per-country bars and the
smallholder scatter. Same data sources as the matplotlib equivalents.

Outputs:
* ``per_country_bars_gg.pdf``
* ``smallholder_scatter_gg.pdf``
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

FIGS = Path(__file__).parent.parent / "figs"
FIGS.mkdir(exist_ok=True, parents=True)

OLIVE = "#5b7026"
SIENNA = "#8b3a1f"
PALETTE = {"Planet wins": OLIVE, "S2 wins": SIENNA}


def _load_deltas() -> pd.DataFrame:
    s2_files = [
        f for f in sorted(glob.glob("logs/ftw_official/b7_*.csv")) if "per_country" not in f
    ]
    s2 = pd.concat([pd.read_csv(f) for f in s2_files], ignore_index=True)
    s2 = s2.rename(columns={"countries": "country", "object_level_f1": "obj_f1"})
    s2 = s2.drop_duplicates(subset="country", keep="last")[["country", "pixel_level_iou", "obj_f1"]]
    s2 = s2.rename(columns={"pixel_level_iou": "iou_s2", "obj_f1": "f1_s2"})

    pl = pd.read_csv("logs/fulldata_eval/planet_b3_augmax_full_ws_tta.csv")
    pl = pl[["country", "pixel_level_iou", "object_ws_f1"]].rename(
        columns={"pixel_level_iou": "iou_pl", "object_ws_f1": "f1_pl"}
    )
    m = s2.merge(pl, on="country", how="inner").copy()
    m["d_iou"] = (m.iou_pl - m.iou_s2) * 100.0
    m["d_f1"] = (m.f1_pl - m.f1_s2) * 100.0
    m["country_lbl"] = m.country.str.replace("_", " ").str.title()
    return m


def per_country_bars(df: pd.DataFrame) -> None:
    order = df.sort_values("d_f1").country_lbl.tolist()

    def _melt(metric_col: str, label: str) -> pd.DataFrame:
        sub = df[["country_lbl", metric_col]].copy()
        sub["metric"] = label
        sub["value"] = sub[metric_col]
        sub["winner"] = np.where(sub["value"] >= 0, "Planet wins", "S2 wins")
        return sub[["country_lbl", "metric", "value", "winner"]]

    plot_df = pd.concat(
        [_melt("d_f1", "$\\Delta$ Obj F1 (pp)"), _melt("d_iou", "$\\Delta$ Pixel IoU (pp)")]
    )
    # Clip the IoU panel domain so Kenya/Portugal don't crush everyone else;
    # annotate the clipped bars separately.
    iou_lim_lo, iou_lim_hi = -15, 15
    plot_df["value_clipped"] = plot_df["value"].where(
        ~(
            (plot_df["metric"] == "$\\Delta$ Pixel IoU (pp)")
            & ((plot_df["value"] < iou_lim_lo) | (plot_df["value"] > iou_lim_hi))
        ),
        np.clip(plot_df["value"], iou_lim_lo, iou_lim_hi),
    )

    p = (
        p9.ggplot(plot_df, p9.aes(x="country_lbl", y="value_clipped", fill="winner"))
        + p9.geom_col(width=0.7, color="black", size=0.15)
        + p9.geom_text(
            p9.aes(label="value", y="value_clipped"),
            data=plot_df,
            format_string="{:+.1f}",
            size=6,
            nudge_y=0.4,
            ha="left",
        )
        + p9.geom_hline(yintercept=0, color="black", size=0.4)
        + p9.scale_x_discrete(limits=order)
        + p9.scale_fill_manual(values=PALETTE, guide=p9.guide_legend(title=""))
        + p9.facet_wrap("~metric", scales="free_x")
        + p9.coord_flip()
        + p9.theme_minimal(base_size=8)
        + p9.theme(
            figure_size=(7.4, 4.0),
            panel_grid_major_y=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            axis_title_y=p9.element_blank(),
            strip_text=p9.element_text(weight="bold"),
            legend_position="top",
        )
        + p9.labs(y="", title="FTW-HD (3 m) vs Sentinel-2 PRUE-B7 full (10 m)")
    )
    out = FIGS / "per_country_bars_gg.pdf"
    p.save(out, dpi=200, verbose=False)
    print(f"wrote {out}")


def smallholder_scatter() -> None:
    src = Path("paper/scripts/output/smallholder_scatter.csv")
    if not src.exists():
        print(f"missing {src}; run make_smallholder_scatter.py first; skipping")
        return
    df = pd.read_csv(src).rename(columns={"median_field_size_ha": "ha"})
    df["d_f1"] = df["delta_obj_f1"] * 100.0
    df["log_ha"] = np.log10(df["ha"])
    df["country_lbl"] = df["country"].str.replace("_", " ").str.title()
    df["winner"] = np.where(df["d_f1"] >= 0, "Planet wins", "S2 wins")

    r = np.corrcoef(df["log_ha"], df["d_f1"])[0, 1]

    p = (
        p9.ggplot(df, p9.aes(x="ha", y="d_f1"))
        + p9.geom_hline(yintercept=0, color="black", size=0.3, linetype="dashed")
        + p9.geom_smooth(method="lm", color="#444444", fill="#bbbbbb", size=0.4, alpha=0.25)
        + p9.geom_point(
            p9.aes(fill="winner", size="ha"),
            shape="o",
            color="black",
            stroke=0.3,
            alpha=0.85,
        )
        + p9.geom_text(p9.aes(label="country_lbl"), size=6, nudge_y=0.35, ha="left")
        + p9.scale_x_log10(
            breaks=[0.1, 0.3, 1, 3, 10, 30, 100],
            labels=lambda x: [f"{v:g}" for v in x],
        )
        + p9.scale_fill_manual(values=PALETTE, guide=p9.guide_legend(title=""))
        + p9.scale_size_continuous(range=[2, 6], guide=None)
        + p9.theme_minimal(base_size=8)
        + p9.theme(
            figure_size=(5.6, 3.6),
            panel_grid_minor=p9.element_blank(),
            legend_position="top",
        )
        + p9.labs(
            x="Per-country median FTW field area (ha, log scale)",
            y="$\\Delta$ Obj F1 (pp, Planet $-$ S2)",
            title=f"Where the 3$\\times$ GSD advantage shows up (Pearson $r$={r:+.2f}, n={len(df)})",
        )
    )
    out = FIGS / "smallholder_scatter_gg.pdf"
    p.save(out, dpi=200, verbose=False)
    print(f"wrote {out}")


def main() -> None:
    df = _load_deltas()
    per_country_bars(df)
    smallholder_scatter()


if __name__ == "__main__":
    main()
