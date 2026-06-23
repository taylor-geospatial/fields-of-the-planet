"""Model-free resolution ceiling: how many TRUE field polygons survive as
distinct, separable polygons when represented at a genuine grid resolution.

The fields tile space (touch), so separating two adjacent fields costs >=1 pixel
of boundary. We rasterize the true polygons as instance ids at resolution r,
mark inter-instance (and field/bg) borders as boundary, take each field's
interior, polygonize it, and IoU-match back to the true polygon. A field is
"recovered" at resolution r if its interior survives and IoU>0.5 with truth.
Reports recall by GT area bin for r in {3 m (Planet), 10 m (S2)} -- the best ANY
model on that sensor could do, independent of model/backbone/upsampling.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely.geometry
from rasterio.features import rasterize
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent))

EDGES = (0.5, 2.0)
BINS = ("small", "medium", "large")
GT_ROOT = "data/ftw_polygons_clipped"


def ceiling_recall(country, patch_id, res_m):
    """Return dict bin -> (n_true, n_recovered) for one patch at resolution res_m."""
    gdf = gpd.read_parquet(Path(GT_ROOT) / country / f"{patch_id}.parquet")
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    if len(gdf) == 0:
        return {}
    true_area = (gdf.geometry.area / 1e4).to_numpy()  # ha, UTM
    minx, miny, maxx, maxy = gdf.total_bounds
    w = max(1, int(np.ceil((maxx - minx) / res_m)))
    h = max(1, int(np.ceil((maxy - miny) / res_m)))
    tr = from_origin(minx, maxy, res_m, res_m)
    # Instance-id raster (1..N); pixel center in polygon. Touching fields keep
    # distinct ids, so this is the OPTIMISTIC assignment.
    ids = rasterize(
        ((g, i + 1) for i, g in enumerate(gdf.geometry)),
        out_shape=(h, w),
        transform=tr,
        fill=0,
        dtype="int32",
    )
    # Boundary = a field pixel whose 4-neighborhood is not all the same id
    # (separating two touching fields costs >=1 boundary pixel). Vectorized via
    # shifts so it is O(image), not O(instances*image).
    bnd = np.zeros_like(ids, dtype=bool)
    bnd[:-1, :] |= ids[:-1, :] != ids[1:, :]
    bnd[1:, :] |= ids[1:, :] != ids[:-1, :]
    bnd[:, :-1] |= ids[:, :-1] != ids[:, 1:]
    bnd[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    interior = (ids > 0) & ~bnd
    res = defaultdict(lambda: [0, 0])
    # Per-field recovery: interior pixels of id k -> polygon -> IoU vs true.
    for k in range(1, len(gdf) + 1):
        b = (
            BINS[0]
            if true_area[k - 1] < EDGES[0]
            else (BINS[1] if true_area[k - 1] < EDGES[1] else BINS[2])
        )
        res[b][0] += 1
        mk = (ids == k) & interior
        if not mk.any():
            continue  # field vanished after boundary erosion
        geoms = [
            shapely.geometry.shape(s)
            for s, v in rio_shapes(mk.astype(np.uint8), transform=tr)
            if v == 1
        ]
        if not geoms:
            continue
        pred = max(geoms, key=lambda g: g.area)
        tru = gdf.geometry.iloc[k - 1]
        inter = pred.intersection(tru).area
        iou = inter / (pred.area + tru.area - inter) if (pred.area + tru.area - inter) > 0 else 0
        if iou > 0.5:
            res[b][1] += 1
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--countries", nargs="+", default=["cambodia"])
    p.add_argument("--n", type=int, default=20, help="patches per country")
    p.add_argument("--out-csv", default=None, help="write pooled + per-country recalls here")
    args = p.parse_args()
    pooled = {r: defaultdict(lambda: [0, 0]) for r in (3.0, 10.0)}
    rows = []  # (country, bin, n_true, recall3, recall10)
    for country in args.countries:
        patches = sorted(Path(GT_ROOT, country).glob("*.parquet"))[: args.n]
        agg = {r: defaultdict(lambda: [0, 0]) for r in (3.0, 10.0)}
        for pp in patches:
            for r in (3.0, 10.0):
                for b, (nt, nr) in ceiling_recall(country, pp.stem, r).items():
                    agg[r][b][0] += nt
                    agg[r][b][1] += nr
                    pooled[r][b][0] += nt
                    pooled[r][b][1] += nr
        print(f"\n=== {country} ({len(patches)} patches) — recovery recall @ resolution ===")
        print(f"{'bin':8s} {'3m recall':>12} {'10m recall':>12}   (n_true)")
        for b in BINS:
            a3, a10 = agg[3.0][b], agg[10.0][b]
            r3, r10 = a3[1] / max(a3[0], 1), a10[1] / max(a10[0], 1)
            print(f"{b:8s} {r3:11.1%} {r10:11.1%}   ({a3[0]})")
            rows.append((country, b, a3[0], r3, r10))

    print(f"\n=== POOLED over {len(args.countries)} regions ===")
    print(f"{'bin':8s} {'3m recall':>12} {'10m recall':>12}   (n_true)")
    for b in BINS:
        a3, a10 = pooled[3.0][b], pooled[10.0][b]
        r3, r10 = a3[1] / max(a3[0], 1), a10[1] / max(a10[0], 1)
        print(f"{b:8s} {r3:11.1%} {r10:11.1%}   ({a3[0]})")
        rows.append(("POOLED", b, a3[0], r3, r10))

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["region", "bin", "n_true", "recall_3m", "recall_10m"])
            for country, b, nt, r3, r10 in rows:
                wtr.writerow([country, b, nt, f"{r3:.4f}", f"{r10:.4f}"])
        print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
