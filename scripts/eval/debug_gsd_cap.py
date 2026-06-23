"""Does scoring at the sensor's GENUINE GSD reconcile model PQ_s with the
representation ceiling? The S2 eval upsamples to 512 (~2-5 m), letting the model
score above the 10 m physical limit. Here we cap each sensor's PREDICTION at its
native GSD (3 m Planet / 10 m S2): rasterize the predicted field polygons onto a
genuine-GSD UTM grid (1-px boundary between touching instances), re-vectorize,
and match the true polygons -- exactly parallel to representation_ceiling.py but
on the model output instead of the GT. Reports small-field recall at the eval
grid vs capped at native GSD.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely.geometry
import torch
from rasterio.features import rasterize
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from shapely.affinity import affine_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_smallfield import _load, _predict_planet, _predict_s2
from polygon_metrics_eval import _eval_grid, _extract_shapes, _true_gt_shapes

from ftw_planet.datasets import FTWPlanet

SMALL_HA = 0.5
GT_ROOT = "data/ftw_polygons_clipped"


def _to_utm(polys_pix, eval_tr, eval_crs, utm_crs):
    """Map polygons from eval-grid pixels -> eval_crs -> UTM."""
    mat = [eval_tr.a, eval_tr.b, eval_tr.d, eval_tr.e, eval_tr.c, eval_tr.f]
    geo = [affine_transform(p, mat) for p in polys_pix]
    if not geo:
        return []
    gs = gpd.GeoSeries(geo, crs=eval_crs)
    if str(eval_crs) != str(utm_crs):
        gs = gs.to_crs(utm_crs)
    return list(gs.geometry)


def _cap_to_gsd(polys_utm, gsd):
    """Render the predicted FIELD polygons at genuine GSD and re-vectorize: the
    field union is rasterized at the sensor resolution, so two predicted fields
    merge iff the predicted gap between them is sub-GSD (10 m can't keep a 2 m
    boundary; 3 m can). This is ~identity at the model's native GSD and coarsens
    an upsampled prediction. No artificial per-instance erosion."""
    polys_utm = [p for p in polys_utm if p and not p.is_empty]
    if not polys_utm:
        return []
    minx, miny, maxx, maxy = gpd.GeoSeries(polys_utm).total_bounds
    w = max(1, int(np.ceil((maxx - minx) / gsd)))
    h = max(1, int(np.ceil((maxy - miny) / gsd)))
    tr = from_origin(minx, maxy, gsd, gsd)
    binary = rasterize(
        ((g, 1) for g in polys_utm), out_shape=(h, w), transform=tr, fill=0, dtype="uint8"
    )
    return [shapely.geometry.shape(s) for s, v in rio_shapes(binary, transform=tr) if v == 1]


BINS = (("small", 0.0, 0.5), ("medium", 0.5, 2.0), ("large", 2.0, 1e9))


def _recall_bins(true_polys, true_areas, preds):
    """Per-bin (n_true, n_matched) via greedy IoU>0.5 match against preds."""
    out = {b: [0, 0] for b, _, _ in BINS}
    used = set()
    for i, a in enumerate(true_areas):
        b = next(name for name, lo, hi in BINS if lo <= a < hi)
        out[b][0] += 1
        t = true_polys[i]
        for j, p in enumerate(preds):
            if j in used or not t.intersects(p):
                continue
            inter = t.intersection(p).area
            iou = inter / (t.area + p.area - inter) if (t.area + p.area - inter) > 0 else 0
            if iou > 0.5:
                used.add(j)
                out[b][1] += 1
                break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--country", default="cambodia")
    p.add_argument("--n", type=int, default=12)
    p.add_argument(
        "--ckpt-planet", default="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    )
    p.add_argument("--ckpt-s2", default="logs/best_checkpoints/s2_efnet7_best.ckpt")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_pl, task_s2 = _load(args.ckpt_planet, dev), _load(args.ckpt_s2, dev)
    pl_ds = FTWPlanet(root="data", countries=[args.country], split="test", load_boundaries=True)
    from ftw_tools.training.datasets import FTW

    s2_ds = FTW(
        root="data/ftw",
        countries=[args.country],
        split="test",
        transforms=None,
        load_boundaries=True,
        temporal_options="stacked",
    )
    s2_by_pid = {Path(f["window_a"]).stem: i for i, f in enumerate(s2_ds.filenames)}
    utm = gpd.read_parquet(sorted(Path(GT_ROOT, args.country).glob("*.parquet"))[0]).crs

    # sensor -> bin -> [n_true, matched_evalgrid, matched_capped]
    tot = defaultdict(lambda: {b: [0, 0, 0] for b, _, _ in BINS})
    n = 0
    for pidx in range(len(pl_ds.records)):
        if n >= args.n:
            break
        pid = str(pl_ds.records[pidx]["patch_id"])
        if pid not in s2_by_pid:
            continue
        for name, task, ds, idx, backend, up, gsd in [
            ("Planet@3m", task_pl, pl_ds, pidx, "planet", None, 3.0),
            ("S2@10m", task_s2, s2_ds, s2_by_pid[pid], "s2", 512, 10.0),
        ]:
            inst = (
                _predict_planet(task, ds, idx, dev)
                if backend == "planet"
                else _predict_s2(task, ds, idx, dev)
            )
            _, ecrs, etr, _ = _eval_grid(ds, idx, args.country, backend, "data", "a", up)
            tp, ta = _true_gt_shapes(GT_ROOT, args.country, pid, ecrs, etr)
            pred_utm = _to_utm(_extract_shapes((inst > 0).astype(np.uint8)), etr, ecrs, utm)
            true_utm = _to_utm(tp, etr, ecrs, utm)
            rec_eval = _recall_bins(true_utm, ta, pred_utm)
            rec_cap = _recall_bins(true_utm, ta, _cap_to_gsd(pred_utm, gsd))
            for b, _, _ in BINS:
                tot[name][b][0] += rec_eval[b][0]
                tot[name][b][1] += rec_eval[b][1]
                tot[name][b][2] += rec_cap[b][1]
        n += 1
    print(f"\n{args.country} ({n} patches) — recall by bin: eval-grid -> @native GSD")
    for k in tot:
        parts = []
        for b, _, _ in BINS:
            nt, me, mc = tot[k][b]
            parts.append(f"{b} {me / max(nt, 1):.0%}->{mc / max(nt, 1):.0%}(n{nt})")
        print(f"  {k:10s} " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
