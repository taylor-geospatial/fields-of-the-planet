"""Worked-example figure for the polygon metrics (Fig. metric_example).

Renders a single PlanetScope patch as [image | GT field instances | predicted
field instances] and prints the per-patch polygon metrics next to it, so the
reader can see what object F1 / PQ / SQ / boundary chamfer actually measure.

Inference (FTP-PRUE+) and the metric computation reuse the
exact functions from ``scripts/eval`` so the example numbers match the paper's
evaluation protocol. Runs on CPU; one patch is a few seconds.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tg_style
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from polygon_metrics_eval import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    GSD_M,
    _boundary_pixels,
    _extract_shapes,
    _match_shapes,
    _symmetric_chamfer,
)
from postprocess_eval import (  # noqa: E402
    _pad_min32,
    _predict_tta,
    watershed_instances,
)

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times"]})
NORM_DIVISOR = 3000.0


def _f1(tps: int, fps: int, fns: int) -> float:
    p = tps / max(tps + fps, 1)
    r = tps / max(tps + fns, 1)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def polys_and_metrics(gt_bin: np.ndarray, pred_bin: np.ndarray, gsd_m: float) -> dict:
    """Vectorize the GT and predicted field masks into polygons (the same
    ``rasterio.features.shapes`` -> shapely step the eval uses), match them by
    IoU, and return the polygons + the per-patch metrics. ``tp_js`` is the set
    of predicted-polygon indices matched to a GT polygon at IoU >= 0.5."""
    import rasterio.features

    gt_shapes = _extract_shapes(gt_bin)
    pred_shapes = _extract_shapes(pred_bin)
    m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)

    t05 = AP_IOU_THRESHOLDS[0]
    tps, fps, fns = m["per_t"][t05]
    rq = _f1(tps, fps, fns)
    matched = m["matched_pairs_low"]
    sq = float(np.mean([iou for _, _, iou in matched])) if matched else 0.0
    pq = sq * rq
    ap = float(np.mean([_f1(*m["per_t"][t]) for t in AP_IOU_THRESHOLDS]))

    chamfers = []
    for i, j, _ in matched:
        pb = _boundary_pixels(
            rasterio.features.rasterize([pred_shapes[j]], out_shape=pred_bin.shape, dtype=np.uint8)
        )
        gb = _boundary_pixels(
            rasterio.features.rasterize([gt_shapes[i]], out_shape=gt_bin.shape, dtype=np.uint8)
        )
        c = _symmetric_chamfer(pb, gb)
        if c is not None:
            chamfers.append(c)
    bnd_m = float(np.mean(chamfers)) * gsd_m if chamfers else float("nan")

    inter = int((pred_bin & gt_bin).sum())
    union = int((pred_bin | gt_bin).sum())
    pix_iou = inter / union if union else 0.0

    return {
        "gt_shapes": gt_shapes,
        "pred_shapes": pred_shapes,
        "tp_js": {j for _, j, _ in matched},
        "n_gt": len(gt_shapes),
        "n_pred": len(pred_shapes),
        "objf1": rq,
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "ap": ap,
        "bnd_m": bnd_m,
        "dN": abs(len(pred_shapes) - len(gt_shapes)),
        "pix_iou": pix_iou,
    }


def patch_metrics(seg_np: np.ndarray, gt_np: np.ndarray, dist: np.ndarray, gsd_m: float) -> dict:
    """Per-patch metrics on the field class, plus the watershed instance raster
    and GT field coverage (used to pick densely-tiled patches)."""
    inst_pred = watershed_instances(seg_np, dist, h_min=2.0, field_class=1)
    res = polys_and_metrics((gt_np == 1).astype(np.uint8), (inst_pred > 0).astype(np.uint8), gsd_m)
    valid = int((gt_np != 3).sum())
    res["coverage"] = int((gt_np == 1).sum()) / valid if valid else 0.0
    res["inst_pred"] = inst_pred
    return res


def _square_crop(arr: np.ndarray) -> np.ndarray:
    """Center-crop a (H,W[,C]) array to its largest centered square, so figure
    panels are uniform and tight (no letterbox whitespace)."""
    h, w = arr.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return arr[top : top + s, left : left + s]


def _inst_cmap(n: int, seed: int = 0) -> mpl.colors.ListedColormap:
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    rng = np.random.default_rng(seed)
    cols = base[rng.integers(0, 20, size=max(n, 1))]
    cols = np.clip(cols + rng.uniform(-0.08, 0.08, cols.shape), 0, 1)
    # Background (label 0) is a soft ivory, not stark white, so any non-field
    # area recedes instead of punching holes in the panel.
    bg = np.array(mpl.colors.to_rgb(tg_style.IVORY))
    return mpl.colors.ListedColormap(np.vstack([bg, cols]))


def show_instances(ax, inst: np.ndarray, title: str) -> None:
    n = int(inst.max())
    remap = np.zeros(inst.max() + 1, dtype=np.int32)
    remap[1:] = np.arange(1, n + 1)
    ax.imshow(remap[inst], cmap=_inst_cmap(n, seed=n), interpolation="nearest")
    ax.set_title(title, fontsize=6.2, color=tg_style.BROWN, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_polygons(ax, res: dict, size: int, title: str) -> None:
    """Draw the vectorized polygons: GT as thin gray outlines, predicted
    polygons filled by match status (green = TP matched to a GT parcel,
    red = FP unmatched). This is the geopandas representation the object
    metrics are actually computed on."""
    import geopandas as gpd

    if res["gt_shapes"]:
        gpd.GeoSeries(res["gt_shapes"]).boundary.plot(ax=ax, color="0.35", linewidth=0.5)
    tp = [s for j, s in enumerate(res["pred_shapes"]) if j in res["tp_js"]]
    fp = [s for j, s in enumerate(res["pred_shapes"]) if j not in res["tp_js"]]
    if tp:
        gpd.GeoSeries(tp).plot(
            ax=ax, facecolor=tg_style.GREEN_INK, edgecolor="#2f5d23", linewidth=0.5, alpha=0.6
        )
    if fp:
        gpd.GeoSeries(fp).plot(
            ax=ax, facecolor=tg_style.RED, edgecolor="#7d2414", linewidth=0.5, alpha=0.6
        )
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=6.2, color=tg_style.BROWN, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=REPO / "logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt",
    )
    ap.add_argument("--root", type=str, default=str(REPO / "data"))
    ap.add_argument("--country", type=str, default="croatia")
    ap.add_argument("--max-patches", type=int, default=140)
    ap.add_argument("--n-show", type=int, default=2)
    ap.add_argument(
        "--idxs",
        type=int,
        nargs="*",
        default=[87, 71],
        help="Explicit patch indices to show (in order). Overrides quantile pick. "
        "Default is the croatia (strong-but-imperfect, weak) pair used in the paper.",
    )
    ap.add_argument("--out", type=Path, default=REPO / "paper/figs/metric_example.pdf")
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.55,
        help="Min GT field-interior fraction; keeps panels densely tiled (no voids).",
    )
    args = ap.parse_args()

    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    task = task.eval()
    model = task.model
    gsd_m = GSD_M["planet"]

    ds = FTWPlanet(
        root=args.root,
        countries=[args.country],
        split="test",
        transforms=None,
        load_boundaries=True,
    )

    cands = []
    for idx in range(min(args.max_patches, len(ds))):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).float() / PLANET_SR_SCALE
        mask = sample["mask"].unsqueeze(0)
        image, mask, H, W = _pad_min32(image, mask, min_size=512, pad_mode="zero")
        probs, _ = _predict_tta(task, model, image, 20.0)
        seg_np = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        boundary_np = (seg_np == 2).astype(np.uint8)
        dist = distance_transform_edt(boundary_np == 0).astype(np.float32)
        met = patch_metrics(seg_np, gt_np, dist, gsd_m)

        # RGB from window-B channels [B,G,R,NIR] -> display [R,G,B].
        rgb = image[0, [2, 1, 0]].cpu().numpy()[:, :H, :W] * PLANET_SR_SCALE
        rgb = np.clip(np.transpose(rgb, (1, 2, 0)) / NORM_DIVISOR, 0, 1)

        # Need enough fields and a finite boundary error. No upper n_gt cap:
        # the densest (best-looking) patches have the most parcels.
        if met["n_gt"] < 8 or not np.isfinite(met["bnd_m"]):
            continue
        cands.append({"idx": idx, "rgb": rgb, "gt": gt_np, "met": met})
        print(
            f"  cand idx={idx} n_gt={met['n_gt']} cov={met['coverage']:.2f} "
            f"objf1={met['objf1']:.3f} pq={met['pq']:.3f}"
        )

    n_show = args.n_show
    if len(cands) < n_show:
        raise SystemExit(
            f"only {len(cands)} candidates in first {args.max_patches} of {args.country}"
        )

    # Pick patches spanning the recognition range (a strong case and a harder
    # one) so the figure shows how the metrics move with quality. Explicit
    # --idxs overrides for a reproducible, hand-picked pair.
    by_idx = {c["idx"]: c for c in cands}
    if args.idxs:
        missing = [i for i in args.idxs if i not in by_idx]
        if missing:
            raise SystemExit(f"requested idxs not among candidates: {missing}")
        picks = [by_idx[i] for i in args.idxs]
    else:
        # Only densely-tiled patches (no forest voids), then take the strongest
        # and the weakest of those for a clear, full-panel quality contrast.
        dense = sorted(
            (c for c in cands if c["met"]["coverage"] >= args.min_coverage),
            key=lambda c: c["met"]["objf1"],
        )
        if len(dense) < n_show:
            raise SystemExit(
                f"only {len(dense)} patches with coverage>={args.min_coverage}; "
                f"lower --min-coverage or raise --max-patches"
            )
        picks = (
            [dense[-1], dense[0]]
            if n_show == 2
            else dense[:: max(1, len(dense) // n_show)][:n_show]
        )
        print(
            f"  selected idxs={[p['idx'] for p in picks]} "
            f"cov={[round(p['met']['coverage'], 2) for p in picks]} "
            f"objf1={[round(p['met']['objf1'], 2) for p in picks]}"
        )

    # Single-column figure. Per row: PlanetScope image, the predicted raster
    # field mask, and that mask vectorized into polygons (geopandas) matched to
    # GT -- so the reader sees the raster->polygon step the metrics run on. One
    # bordered metrics table spans both rows. Polygons + table are recomputed on
    # the displayed square crop so image, mask, polygons, and numbers all agree.
    titles = ["PlanetScope", "Pred mask", "Polygons (TP/FP)"]
    fig = plt.figure(figsize=(3.5, 0.82 * n_show + 0.14))
    gs = fig.add_gridspec(
        n_show,
        4,
        width_ratios=[1, 1, 1, 1.55],
        wspace=0.04,
        hspace=0.04,
        left=0.004,
        right=0.996,
        top=0.86,
        bottom=0.02,
    )
    row_res = []
    for r, pick in enumerate(picks):
        inst_c = _square_crop(pick["met"]["inst_pred"])
        gt_bin_c = _square_crop((pick["gt"] == 1).astype(np.uint8))
        pred_bin_c = (inst_c > 0).astype(np.uint8)
        size = gt_bin_c.shape[0]
        res = polys_and_metrics(gt_bin_c, pred_bin_c, gsd_m)
        row_res.append(res)

        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(_square_crop(pick["rgb"]))
        ax0.set_xticks([])
        ax0.set_yticks([])
        if r == 0:
            ax0.set_title(titles[0], fontsize=6.2, color=tg_style.BROWN, pad=1.5)
        show_instances(fig.add_subplot(gs[r, 1]), inst_c, titles[1] if r == 0 else "")
        plot_polygons(fig.add_subplot(gs[r, 2]), res, size, titles[2] if r == 0 else "")

    # Bordered metrics table (rows = metrics, one value column per example).
    # [0,1] metrics shown x100 at 1 decimal (matching the paper tables);
    # chamfer in meters; |dN| an integer count.
    metric_rows = [
        ("pixel IoU", "pix_iou", "{:.1f}", 100.0),
        ("Obj F1@.5", "objf1", "{:.1f}", 100.0),
        ("PQ", "pq", "{:.1f}", 100.0),
        ("SQ", "sq", "{:.1f}", 100.0),
        ("RQ", "rq", "{:.1f}", 100.0),
        ("F1 .5:.95", "ap", "{:.1f}", 100.0),
        ("chamfer m", "bnd_m", "{:.1f}", 1.0),
        ("|dN|", "dN", "{:d}", 1.0),
    ]

    def _fmt(value: float, spec: str, scale: float) -> str:
        v = value * scale
        return spec.format(round(v)) if spec == "{:d}" else spec.format(v)

    cell_text = [
        [_fmt(res[key], spec, scale) for res in row_res] for (_, key, spec, scale) in metric_rows
    ]
    row_labels = [name for name, _, _, _ in metric_rows]
    col_labels = (["top", "bot"] if n_show == 2 else [f"#{i + 1}" for i in range(n_show)])[:n_show]

    ax_t = fig.add_subplot(gs[:, 3])
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0.42, 0.0, 0.58, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.3)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor(tg_style.BROWN)
        cell.set_linewidth(0.4)
        cell.get_text().set_color(tg_style.BROWN)
        cell.get_text().set_ha("center")
        if rr == 0:
            cell.set_facecolor("#efece0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}  (country={args.country}, idxs={[p['idx'] for p in picks]})")
    for p in picks:
        print(
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in p["met"].items()
                if k != "inst_pred"
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
