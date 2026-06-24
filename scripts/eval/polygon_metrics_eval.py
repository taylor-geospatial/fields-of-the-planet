"""GSD-aware polygon-level metrics for field-boundary segmentation.

Per checkpoint, per country, computes:

* **PQ / SQ / RQ** (panoptic quality, IoU>=0.5 matches)
* **F1@[0.5:0.05:0.95]** (mean object F1 over IoU thresholds, single-pass)
* **Polygon count delta** (|N_pred - N_gt|) per patch, mean+median
* **Boundary error (m)** symmetric chamfer * GSD on IoU>=0.5 matches,
  mean+p95

Reuses the inference/watershed pipeline from ``scripts/postprocess_eval.py``
so numbers are comparable. Outputs a single CSV row per country.

Example:
    uv run scripts/polygon_metrics_eval.py \\
        --ckpt logs/.../checkpoints/last.ckpt \\
        --out logs/polygon_metrics/foo.csv \\
        --dataset-backend planet --min-pad-size 512 \\
        --watershed --tta
"""

import argparse
import time
from pathlib import Path
from typing import TypedDict

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask

# Reuse all the inference + watershed infrastructure from postprocess_eval.
from postprocess_eval import (
    COUNTRIES,
    _pad_min32,
    _predict,
    _predict_tta,
    watershed_instances,
)
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.affinity import affine_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

# IoU thresholds for F1@[0.5:0.05:0.95]; ``ap_5_95`` is the legacy CSV column.
AP_IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2).tolist()

# Ground-sample distance in meters (Planet 3m, S2 10m).
GSD_M = {"planet": 3.0, "s2": 10.0}


def _extract_shapes(mask: np.ndarray) -> list[shapely.geometry.base.BaseGeometry]:
    """Connected-component shapes of value==1 in ``mask``.

    Matches the shape extraction used by ``get_object_level_metrics``.
    """
    shapes = []
    for geom, val in rasterio.features.shapes(mask.astype(np.uint8)):
        if val == 1:
            shapes.append(shapely.geometry.shape(geom))
    return shapes


def _true_gt_shapes(
    polygons_root: str,
    country: str,
    patch_id: str,
    eval_crs,
    eval_transform,
) -> tuple[list[shapely.geometry.base.BaseGeometry], list[float]]:
    """True FTW field polygons for a patch, mapped into the eval pixel grid.

    Reads the per-patch GeoParquet written by
    ``scripts/pipeline/clip_polygons_per_patch.py`` (geometries in the patch's
    Planet UTM CRS, clipped to bounds), reprojects to the eval grid CRS, and
    applies the inverse of the eval raster affine so coordinates land in the
    same (col, row) pixel frame as the predicted shapes from ``_extract_shapes``.
    Backend-agnostic: the same true parcels are mapped into each sensor's eval
    grid, so Planet and S2 are scored against identical ground truth. Returns
    ``(shapes_in_pixel_coords, true_area_ha_per_shape)``; the true areas come
    from the UTM parquet so size bins are grid-independent.
    """
    ppath = Path(polygons_root) / country / f"{patch_id}.parquet"
    if not ppath.exists():
        return [], []
    gdf = gpd.read_parquet(ppath).explode(index_parts=False, ignore_index=True)
    # True planimetric area (ha) in the parquet's UTM CRS -- grid-independent, so
    # area bins are consistent across sensors (unlike pixel-area on the eval grid).
    true_area_ha = (gdf.geometry.area / 1e4).tolist()
    gdf_eval = gdf
    if gdf.crs is not None and eval_crs is not None and str(gdf.crs) != str(eval_crs):
        gdf_eval = gdf.to_crs(eval_crs)
    inv = ~eval_transform
    matrix = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]
    shapes: list[shapely.geometry.base.BaseGeometry] = []
    areas: list[float] = []
    for geom, area_ha in zip(gdf_eval.geometry, true_area_ha, strict=True):
        if geom is None or geom.is_empty:
            continue
        shapes.append(affine_transform(geom, matrix))
        areas.append(area_ha)
    return shapes, areas


def _planet_pixel_area_ha(root, country, patch_id, window):
    """True ground area (ha) of one Planet patch pixel, from its UTM raster."""
    tif = Path(root) / "planet" / country / f"window_{window}" / f"{patch_id}.tif"
    with rasterio.open(tif) as src:
        return abs(src.transform.a * src.transform.e) / 1e4, src.width, src.height


def _eval_grid(ds, idx, country, dataset_backend, root, planet_window, upsample_to):
    """(patch_id, eval_crs, eval_transform, pixel_area_ha) for dataset index ``idx``.

    The eval transform maps eval-grid pixels -> geo, accounting for the S2
    256->upsample_to interpolation so true polygons land on the prediction grid.
    ``pixel_area_ha`` is the TRUE ground area of one eval-grid pixel (from the
    Planet patch's metric extent), so predicted-FP area bins are correct on both
    sensors; None if the Planet patch is unavailable (caller falls back).
    """
    if dataset_backend == "planet":
        patch_id = str(ds.records[idx]["patch_id"])
        px_ha, _, _ = _planet_pixel_area_ha(root, country, patch_id, planet_window)
        tif = Path(root) / "planet" / country / f"window_{planet_window}" / f"{patch_id}.tif"
        with rasterio.open(tif) as src:
            return patch_id, src.crs, src.transform, px_ha
    fn = ds.filenames[idx]["window_a"]
    patch_id = Path(fn).stem
    with rasterio.open(fn) as src:
        crs, tr, sw, sh = src.crs, src.transform, src.width, src.height
    eval_px = upsample_to if upsample_to is not None else sw
    if upsample_to is not None:
        tr = tr * rasterio.Affine.scale(sw / upsample_to, sh / upsample_to)
    # True pixel area for the S2 eval grid: the patch's metric area (from the
    # co-registered Planet patch) spread over the square eval grid.
    ppath = Path(root) / "planet" / country / f"window_{planet_window}" / f"{patch_id}.tif"
    px_ha = None
    if ppath.exists():
        ppx_ha, pw, ph = _planet_pixel_area_ha(root, country, patch_id, planet_window)
        px_ha = (ppx_ha * pw * ph) / (eval_px * eval_px)
    return patch_id, crs, tr, px_ha


def _true_gt_utm(polygons_root, country, patch_id):
    """True FTW polygons in their UTM CRS (exploded) + area_ha (UTM) + the CRS.
    Used for native-GSD scoring, where matching is done in metric UTM space."""
    ppath = Path(polygons_root) / country / f"{patch_id}.parquet"
    if not ppath.exists():
        return [], [], None
    gdf = gpd.read_parquet(ppath).explode(index_parts=False, ignore_index=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    return list(gdf.geometry), (gdf.geometry.area / 1e4).tolist(), gdf.crs


def _cap_pred_to_gsd(pred_pixel_shapes, eval_transform, eval_crs, utm_crs, gsd):
    """Render the predicted field polygons at a genuine ground resolution and
    re-vectorize. Map eval-grid pixel polygons -> eval_crs -> UTM, rasterize the
    field union at ``gsd`` m, take connected components: two predicted fields
    merge iff the gap between them is sub-GSD (10 m cannot keep a 2 m boundary;
    3 m can). ~identity at the model's native GSD, coarsens an upsampled one.
    Returns polygons in UTM metres."""
    if not pred_pixel_shapes:
        return []
    mat = [
        eval_transform.a,
        eval_transform.b,
        eval_transform.d,
        eval_transform.e,
        eval_transform.c,
        eval_transform.f,
    ]
    gs = gpd.GeoSeries([affine_transform(p, mat) for p in pred_pixel_shapes], crs=eval_crs)
    if utm_crs is not None and str(eval_crs) != str(utm_crs):
        gs = gs.to_crs(utm_crs)
    polys = [p for p in gs.geometry if p is not None and not p.is_empty]
    if not polys:
        return []
    minx, miny, maxx, maxy = gpd.GeoSeries(polys).total_bounds
    w = max(1, int(np.ceil((maxx - minx) / gsd)))
    h = max(1, int(np.ceil((maxy - miny) / gsd)))
    tr = from_origin(minx, maxy, gsd, gsd)
    binary = rasterio.features.rasterize(
        ((p, 1) for p in polys), out_shape=(h, w), transform=tr, fill=0, dtype="uint8"
    )
    return [
        shapely.geometry.shape(s)
        for s, v in rasterio.features.shapes(binary, transform=tr)
        if v == 1
    ]


AREA_BIN_LABELS = ("small", "medium", "large")


class _BinCell(TypedDict):
    """Per-(area-bin, IoU-threshold) tallies: integer match counts + matched IoUs."""

    tp: int
    fp: int
    fn: int
    ious: list[float]


def _new_bin_cell() -> _BinCell:
    return {"tp": 0, "fp": 0, "fn": 0, "ious": []}


def _area_bin(area_ha: float, edges: tuple) -> str:
    """Bin a polygon by area (ha). edges=(0.5,2.0) -> small<0.5, medium 0.5-2, large>2."""
    for k, e in enumerate(edges):
        if area_ha < e:
            return AREA_BIN_LABELS[k]
    return AREA_BIN_LABELS[len(edges)]


def _match_shapes(
    gt_shapes: list,
    pred_shapes: list,
    iou_thresholds: list[float],
) -> dict:
    """Greedy-match GT vs predicted shapes once, return per-threshold tps/fps/fns
    plus the matched IoUs and pair indices at the lowest threshold.

    For a fixed greedy strategy (the same one ``get_object_level_metrics``
    uses — first prediction with IoU>thresh wins), tps/fps/fns at every
    threshold can be derived from a single pairwise IoU pass. This keeps
    AP@[0.5:0.95] cheap.
    """
    n_gt, n_pred = len(gt_shapes), len(pred_shapes)
    # Precompute IoUs for all intersecting pairs.
    ious = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, g in enumerate(gt_shapes):
        for j, p in enumerate(pred_shapes):
            if not g.intersects(p):
                continue
            inter = g.intersection(p).area
            if inter <= 0:
                continue
            union = g.area + p.area - inter
            if union <= 0:
                continue
            ious[i, j] = inter / union

    per_t = {}
    pairs_per_t: dict[float, list] = {}  # matched (i,j,iou) at each threshold
    for t in iou_thresholds:
        matched_j: set[int] = set()
        pairs: list[tuple[int, int, float]] = []
        for i in range(n_gt):
            for j in range(n_pred):
                if j in matched_j:
                    continue
                iou = float(ious[i, j])
                if iou > t:
                    matched_j.add(j)
                    pairs.append((i, j, iou))
                    break
        per_t[t] = (len(pairs), n_pred - len(matched_j), n_gt - len(pairs))
        pairs_per_t[t] = pairs
    return {
        "per_t": per_t,
        "matched_pairs_low": pairs_per_t[iou_thresholds[0]],  # SQ + chamfer
        "pairs_per_t": pairs_per_t,
    }


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Boolean array of 1-pixel-thick boundary of ``mask`` (uint8 0/1)."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    # Boundary = pixels in mask whose distance-to-background is < 1.5
    # (4-connected boundary). Faster than morphological gradient.
    dt = distance_transform_edt(mask)
    return (mask.astype(bool)) & (dt < 1.5)


def _symmetric_chamfer(pred_bd: np.ndarray, gt_bd: np.ndarray) -> float | None:
    """Mean symmetric chamfer (pixels) between two boundary masks."""
    if not pred_bd.any() or not gt_bd.any():
        return None
    dt_pred = distance_transform_edt(~pred_bd)
    dt_gt = distance_transform_edt(~gt_bd)
    d1 = float(dt_pred[gt_bd].mean())
    d2 = float(dt_gt[pred_bd].mean())
    return 0.5 * (d1 + d2)


def evaluate_country(
    task,
    model,
    device,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    use_tta: bool,
    use_watershed: bool,
    h_min: float,
    sdf_clip: float,
    min_pad_size: int,
    pad_mode: str,
    dataset_backend: str,
    s2_data_scale: float,
    upsample_to: int | None,
    area_edges: tuple | None = None,
    pixel_area_ha: float = 0.0,
    bin_stats: dict[str, dict[float, _BinCell]] | None = None,
    gt_polygons_root: str | None = None,
    planet_window: str = "a",
    score_gsd_m: float | None = None,
) -> dict[str, float]:
    if dataset_backend == "s2":
        from ftw_tools.training.datasets import FTW

        ds = FTW(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
            temporal_options="stacked",
        )
        scale = float(s2_data_scale)
    else:
        ds = FTWPlanet(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
        )
        scale = PLANET_SR_SCALE
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    gsd_m = GSD_M[dataset_backend]

    # Per-threshold counts (for AP and PQ).
    counts = {t: [0, 0, 0] for t in AP_IOU_THRESHOLDS}  # tps, fps, fns
    matched_ious_05: list[float] = []  # for SQ
    chamfer_pixels: list[float] = []  # for boundary error
    polygon_deltas: list[int] = []
    n_pred_per_patch: list[int] = []
    n_gt_per_patch: list[int] = []
    n_patches = 0

    for idx, batch in enumerate(tqdm(dl, desc=country, leave=False)):
        image = batch["image"].to(device) / scale
        mask = batch["mask"].to(device)
        if upsample_to is not None:
            # Resized-S2 control: bilinear-upsample image, nearest-upsample mask
            # 256->upsample_to to match the upsampled-S2 training resolution.
            image = torch.nn.functional.interpolate(
                image, size=(upsample_to, upsample_to), mode="bilinear", align_corners=False
            )
            mask = (
                torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(), size=(upsample_to, upsample_to), mode="nearest"
                )
                .squeeze(1)
                .to(mask.dtype)
            )
        image, mask, H, W = _pad_min32(image, mask, min_size=min_pad_size, pad_mode=pad_mode)

        if use_tta:
            probs, sdf = _predict_tta(task, model, image, sdf_clip)
        else:
            probs, sdf = _predict(task, model, image, sdf_clip)
        seg_pred = probs.argmax(dim=1)  # (1,H,W) 0/1/2

        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        mask_eval[mask == 3] = 3

        seg_np = seg_pred.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        if use_watershed:
            if sdf is not None:
                dist = sdf.squeeze(0).cpu().numpy().astype(np.float32)[:H, :W]
            else:
                boundary_np = (seg_np == 2).astype(np.uint8)
                dist = distance_transform_edt(boundary_np == 0).astype(np.float32)
            inst_pred = watershed_instances(seg_np, dist, h_min=h_min, field_class=1)
            pred_bin = (inst_pred > 0).astype(np.uint8)
        else:
            pred_bin = (seg_np == 1).astype(np.uint8)

        gt_true_areas = None
        patch_pixel_area_ha = pixel_area_ha
        gsd_mode = score_gsd_m is not None and gt_polygons_root is not None
        if gsd_mode:
            # Native-GSD scoring: match in metric UTM space, with the prediction
            # rendered at the sensor's true ground resolution (caps "super-resolved"
            # upsampled S2 back to 10 m). GT = true polygons (UTM), binned by true area.
            patch_id, eval_crs, eval_tr, _ = _eval_grid(
                ds, idx, country, dataset_backend, root, planet_window, upsample_to
            )
            gt_shapes, gt_true_areas, utm_crs = _true_gt_utm(gt_polygons_root, country, patch_id)
            pred_shapes = _cap_pred_to_gsd(
                _extract_shapes(pred_bin), eval_tr, eval_crs, utm_crs, score_gsd_m
            )
            patch_pixel_area_ha = 1e-4  # pred shapes are UTM m^2 -> * 1e-4 = ha
        elif gt_polygons_root is not None:
            # True FTW vector parcels mapped into this sensor's eval grid, not
            # connected components of the rasterized GT mask.
            patch_id, eval_crs, eval_tr, corr_px_ha = _eval_grid(
                ds, idx, country, dataset_backend, root, planet_window, upsample_to
            )
            gt_shapes, gt_true_areas = _true_gt_shapes(
                gt_polygons_root, country, patch_id, eval_crs, eval_tr
            )
            if corr_px_ha is not None:
                patch_pixel_area_ha = corr_px_ha
            pred_shapes = _extract_shapes(pred_bin)
        else:
            gt_bin = (gt_np == 1).astype(np.uint8)
            gt_shapes = _extract_shapes(gt_bin)
            pred_shapes = _extract_shapes(pred_bin)

        n_patches += 1
        n_pred_per_patch.append(len(pred_shapes))
        n_gt_per_patch.append(len(gt_shapes))
        polygon_deltas.append(abs(len(pred_shapes) - len(gt_shapes)))

        m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)
        for t, (tps, fps, fns) in m["per_t"].items():
            counts[t][0] += tps
            counts[t][1] += fps
            counts[t][2] += fns

        # SQ + chamfer use the IoU>=0.5 matches. In gsd_mode the shapes live in
        # metric UTM space (not pixels), so the pixel-grid chamfer is skipped; SQ
        # (matched IoU, scale-free) is still accumulated.
        for i, j, iou in m["matched_pairs_low"]:
            matched_ious_05.append(iou)
            if gsd_mode:
                continue
            # Rasterize each matched shape into a tight bbox and chamfer.
            pred_mask = rasterio.features.rasterize(
                [pred_shapes[j]], out_shape=pred_bin.shape, dtype=np.uint8
            )
            gt_mask = rasterio.features.rasterize(
                [gt_shapes[i]], out_shape=pred_bin.shape, dtype=np.uint8
            )
            pb = _boundary_pixels(pred_mask)
            gb = _boundary_pixels(gt_mask)
            c = _symmetric_chamfer(pb, gb)
            if c is not None:
                chamfer_pixels.append(c)

        # ---- per-area-bin x per-IoU-threshold accumulation ----
        # GT binned by area (ha); predicted FPs by predicted area. With true-GT,
        # GT uses true UTM polygon area (grid-independent) and preds use the
        # patch's true pixel area, so bins are consistent across sensors.
        if bin_stats is not None:
            # bin_stats and area_edges are set together by the caller.
            assert area_edges is not None
            if gt_true_areas is not None:
                gt_bins = [_area_bin(a, area_edges) for a in gt_true_areas]
            else:
                gt_bins = [_area_bin(g.area * pixel_area_ha, area_edges) for g in gt_shapes]
            pred_bins = [_area_bin(p.area * patch_pixel_area_ha, area_edges) for p in pred_shapes]
            for t in AP_IOU_THRESHOLDS:
                matched_gt = {i: iou for i, j, iou in m["pairs_per_t"][t]}
                matched_pred = {j for _, j, _ in m["pairs_per_t"][t]}
                for i, b in enumerate(gt_bins):
                    if i in matched_gt:
                        bin_stats[b][t]["tp"] += 1
                        bin_stats[b][t]["ious"].append(matched_gt[i])
                    else:
                        bin_stats[b][t]["fn"] += 1
                for j, b in enumerate(pred_bins):
                    if j not in matched_pred:
                        bin_stats[b][t]["fp"] += 1

    # ---- Aggregate ----
    def _f1(tps: int, fps: int, fns: int) -> float:
        p = tps / max(tps + fps, 1)
        r = tps / max(tps + fns, 1)
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    # PQ at IoU>=0.5
    t05 = AP_IOU_THRESHOLDS[0]
    tps, fps, fns = counts[t05]
    rq = _f1(tps, fps, fns)
    sq = float(np.mean(matched_ious_05)) if matched_ious_05 else 0.0
    pq = sq * rq
    ap = float(np.mean([_f1(*counts[t]) for t in AP_IOU_THRESHOLDS]))

    chamfer_m = (
        np.array(chamfer_pixels, dtype=np.float64) * gsd_m
        if chamfer_pixels
        else np.array([], dtype=np.float64)
    )

    return {
        "n_patches": n_patches,
        "pq_sq": sq,
        "pq_rq": rq,
        "pq": pq,
        "ap_5_95": ap,
        "n_pred_mean": float(np.mean(n_pred_per_patch)) if n_pred_per_patch else 0.0,
        "n_gt_mean": float(np.mean(n_gt_per_patch)) if n_gt_per_patch else 0.0,
        "polygon_count_delta_mean": (float(np.mean(polygon_deltas)) if polygon_deltas else 0.0),
        "polygon_count_delta_median": (float(np.median(polygon_deltas)) if polygon_deltas else 0.0),
        "boundary_error_m_mean": float(chamfer_m.mean()) if chamfer_m.size else float("nan"),
        "boundary_error_m_p95": (
            float(np.percentile(chamfer_m, 95)) if chamfer_m.size else float("nan")
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--countries", nargs="*", default=None)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--watershed", action="store_true")
    p.add_argument("--h-min", type=float, default=2.0)
    p.add_argument("--sdf-clip", type=float, default=20.0)
    p.add_argument("--min-pad-size", type=int, default=0)
    p.add_argument("--pad-mode", type=str, default="zero", choices=["zero", "replicate"])
    p.add_argument("--dataset-backend", type=str, default="planet", choices=["planet", "s2"])
    p.add_argument("--s2-data-scale", type=float, default=3000.0)
    p.add_argument(
        "--upsample-to",
        type=int,
        default=None,
        help="Resized-S2 control: upsample image (bilinear) + mask (nearest) "
        "256->N before padding. Use with --dataset-backend s2 and N==min-pad-size.",
    )
    p.add_argument(
        "--area-bins",
        type=str,
        default=None,
        help="Comma-separated ha edges, e.g. '0.5,2' -> small<0.5 / medium 0.5-2 / "
        "large>2 by GT polygon area. Pooled (micro) PQ/RQ/SQ per bin are written to "
        "<out>.bins.csv. Requires --pixel-size-m.",
    )
    p.add_argument(
        "--pixel-size-m",
        type=float,
        default=None,
        help="Physical pixel size (m) of the eval grid, for area->ha conversion: "
        "planet native 3, s2 native 10, s2 upsample-512 5.",
    )
    p.add_argument(
        "--gt-polygons-root",
        type=str,
        default=None,
        help="Score against the TRUE FTW vector polygons (per-patch parquets from "
        "clip_polygons_per_patch.py) instead of connected components of the "
        "rasterized GT mask. Planet backend only (uses the Planet patch grid).",
    )
    p.add_argument(
        "--planet-window",
        type=str,
        default="a",
        help="Planet window whose raster affine maps true polygons -> pixel grid.",
    )
    p.add_argument(
        "--score-gsd-m",
        type=float,
        default=None,
        help="Score at this genuine ground resolution (m): the prediction is "
        "re-rendered at this GSD before matching, capping 'super-resolved' upsampled "
        "S2 back to its true 10 m. Requires --gt-polygons-root. ~no-op at Planet 3 m.",
    )
    args = p.parse_args()
    area_edges = tuple(float(x) for x in args.area_bins.split(",")) if args.area_bins else None
    pixel_area_ha = (args.pixel_size_m**2) / 1e4 if args.pixel_size_m else 0.0
    bin_stats: dict[str, dict[float, _BinCell]] | None = (
        {b: {t: _new_bin_cell() for t in AP_IOU_THRESHOLDS} for b in AREA_BIN_LABELS}
        if area_edges
        else None
    )
    if area_edges and not args.pixel_size_m:
        p.error("--area-bins requires --pixel-size-m")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(
        f"device={device} ckpt={args.ckpt} backend={args.dataset_backend} "
        f"tta={args.tta} watershed={args.watershed}"
    )

    tic = time.time()
    task = None
    try:
        from ftw_planet.trainers import FrameFieldSegTask

        task = FrameFieldSegTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
        print("loaded as FrameFieldSegTask")
    # A checkpoint saved for a different task class fails to reconstruct: its
    # hparams don't match the constructor (TypeError/KeyError) or the state_dict
    # keys don't align (RuntimeError). Narrow to those so a genuine bug in the
    # matching class still raises instead of being silently swallowed.
    # argparse.ArgumentError: jsonargparse rejects the ckpt's saved class_path
    # when it is not this task class (e.g. a plain FTWPlanetSegTask checkpoint),
    # which is also just a "wrong task class" signal -> fall through.
    except (RuntimeError, KeyError, TypeError, argparse.ArgumentError) as e:
        print(f"not FrameFieldSegTask ({type(e).__name__}); trying next task class")
    if task is None:
        try:
            from ftw_planet.trainers import SDFSegTask

            task = SDFSegTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
            print("loaded as SDFSegTask")
        except (RuntimeError, KeyError, TypeError, argparse.ArgumentError) as e:
            print(f"not SDFSegTask ({type(e).__name__}); falling back to semantic seg")
            task = CustomSemanticSegmentationTask.load_from_checkpoint(
                str(args.ckpt), map_location="cpu"
            )
            print("loaded as CustomSemanticSegmentationTask")
    task = task.eval().to(device)
    model = task.model
    print(f"loaded model in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "train_checkpoint",
        "country",
        "n_patches",
        "pq_sq",
        "pq_rq",
        "pq",
        "ap_5_95",
        "n_pred_mean",
        "n_gt_mean",
        "polygon_count_delta_mean",
        "polygon_count_delta_median",
        "boundary_error_m_mean",
        "boundary_error_m_p95",
    ]
    if not args.out.exists():
        with args.out.open("w") as f:
            f.write(",".join(cols) + "\n")

    countries = args.countries or COUNTRIES
    for country in countries:
        print(f"=== {country} ({args.split}) ===")
        try:
            m = evaluate_country(
                task,
                model,
                device,
                country,
                args.root,
                args.split,
                args.num_workers,
                args.tta,
                args.watershed,
                args.h_min,
                args.sdf_clip,
                min_pad_size=args.min_pad_size,
                pad_mode=args.pad_mode,
                dataset_backend=args.dataset_backend,
                s2_data_scale=args.s2_data_scale,
                upsample_to=args.upsample_to,
                area_edges=area_edges,
                pixel_area_ha=pixel_area_ha,
                bin_stats=bin_stats,
                gt_polygons_root=args.gt_polygons_root,
                planet_window=args.planet_window,
                score_gsd_m=args.score_gsd_m,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  skip {country}: {e}")
            continue

        row = [
            str(args.ckpt),
            country,
            str(m["n_patches"]),
            f"{m['pq_sq']:.6f}",
            f"{m['pq_rq']:.6f}",
            f"{m['pq']:.6f}",
            f"{m['ap_5_95']:.6f}",
            f"{m['n_pred_mean']:.4f}",
            f"{m['n_gt_mean']:.4f}",
            f"{m['polygon_count_delta_mean']:.4f}",
            f"{m['polygon_count_delta_median']:.4f}",
            f"{m['boundary_error_m_mean']:.4f}",
            f"{m['boundary_error_m_p95']:.4f}",
        ]
        with args.out.open("a") as f:
            f.write(",".join(row) + "\n")
        print(
            f"  PQ={m['pq']:.4f} (SQ={m['pq_sq']:.3f}, RQ={m['pq_rq']:.3f}) "
            f"AP={m['ap_5_95']:.4f} bnd_err={m['boundary_error_m_mean']:.2f}m"
        )

    if bin_stats is not None:
        # Pooled (micro) PQ/SQ/RQ per area bin, across area bins AND IoU thresholds.
        t05 = AP_IOU_THRESHOLDS[0]
        t75 = min(AP_IOU_THRESHOLDS, key=lambda x: abs(x - 0.75))

        def _f1(tp: int, fp: int, fn: int) -> float:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        all_bin: dict[float, _BinCell] = {t: _new_bin_cell() for t in AP_IOU_THRESHOLDS}
        for b in AREA_BIN_LABELS:
            for t in AP_IOU_THRESHOLDS:
                all_bin[t]["tp"] += bin_stats[b][t]["tp"]
                all_bin[t]["fp"] += bin_stats[b][t]["fp"]
                all_bin[t]["fn"] += bin_stats[b][t]["fn"]
                all_bin[t]["ious"] += bin_stats[b][t]["ious"]

        bins_out = Path(str(args.out) + ".bins.csv")
        with bins_out.open("w") as f:
            f.write("area_bins,bin,n_gt,n_pred,pq,sq,rq_50,f1_75,ap_5_95\n")
            for label, st in [*((b, bin_stats[b]) for b in AREA_BIN_LABELS), ("all", all_bin)]:
                rq = _f1(st[t05]["tp"], st[t05]["fp"], st[t05]["fn"])
                f1_75 = _f1(st[t75]["tp"], st[t75]["fp"], st[t75]["fn"])
                ap = float(
                    np.mean([_f1(st[t]["tp"], st[t]["fp"], st[t]["fn"]) for t in AP_IOU_THRESHOLDS])
                )
                sq = float(np.mean(st[t05]["ious"])) if st[t05]["ious"] else 0.0
                n_gt = st[t05]["tp"] + st[t05]["fn"]
                n_pred = st[t05]["tp"] + st[t05]["fp"]
                f.write(
                    f"{args.area_bins},{label},{n_gt},{n_pred},{sq * rq:.4f},{sq:.4f},{rq:.4f},{f1_75:.4f},{ap:.4f}\n"
                )
                print(
                    f"  [{label:7s}] n_gt={n_gt:6d} PQ={sq * rq * 100:5.1f} SQ={sq * 100:5.1f} "
                    f"RQ@.5={rq * 100:5.1f} F1@.75={f1_75 * 100:5.1f} AP={ap * 100:5.1f}"
                )
        print(f"wrote {bins_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
