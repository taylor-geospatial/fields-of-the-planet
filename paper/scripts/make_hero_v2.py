"""Hero v2: label-overlay style.

For each picked patch, render two cells side-by-side: Planet RGB, and
Planet RGB with the 3-class label as a translucent overlay (field
interior cream-tinted, boundary outlined in saturated orange). A small
S2 thumbnail occupies the right-hand corner of each pair to provide
the "see what you would have had at 10 m" contrast.

Same candidate-selection logic as ``make_hero.py`` but a denser layout:
default 8 picks arranged as 4 pairs/row x 2 rows = 8 wide cells per row.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.transform import resize

NORM_DIVISOR = 3000.0
SQUARE_PX = 320

# Tinted overlay colors. Cream for interior, orange for boundary. RGBA in [0,1].
OVERLAY_FIELD = np.array([0.94, 0.86, 0.55])
OVERLAY_BOUND = np.array([0.92, 0.45, 0.10])
ALPHA_FIELD = 0.28
ALPHA_BOUND = 0.85


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ftw-root", type=Path, default=Path("../data/ftw"))
    p.add_argument("--planet-root", type=Path, default=Path("../data/planet"))
    p.add_argument(
        "--udm2-quality", type=Path, default=Path("../data/planet/_global/udm2_quality.jsonl")
    )
    p.add_argument("--out", type=Path, default=Path("hero_v2.pdf"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--pairs-per-row", type=int, default=4)
    p.add_argument("--min-field-pct", type=float, default=0.40)
    p.add_argument("--max-check", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _stretch(rgb, divisor=NORM_DIVISOR):
    out = rgb.astype(np.float32) / divisor
    return np.clip(out, 0.0, 1.0)


def _to_square(img, size=SQUARE_PX):
    order = 1 if img.ndim == 3 else 0
    return resize(
        img,
        (size, size) if img.ndim == 2 else (size, size, img.shape[-1]),
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0),
    )


def _load_planet_rgb(path, size):
    with rasterio.open(path) as src:
        bands = src.read([3, 2, 1])  # R, G, B
    rgb = np.transpose(bands, (1, 2, 0))
    return _to_square(_stretch(rgb), size)


def _load_label(path, size):
    with rasterio.open(path) as src:
        lbl = src.read(1)
    return _to_square(lbl, size).astype(np.uint8)


def _overlay(rgb, label):
    """Blend translucent class colors over the RGB. Outline boundary class."""
    out = rgb.copy()
    f_mask = label == 1
    b_mask = label == 2
    out[f_mask] = (1.0 - ALPHA_FIELD) * out[f_mask] + ALPHA_FIELD * OVERLAY_FIELD
    out[b_mask] = (1.0 - ALPHA_BOUND) * out[b_mask] + ALPHA_BOUND * OVERLAY_BOUND
    return np.clip(out, 0.0, 1.0)


def _candidates(args):
    rows = []
    with args.udm2_quality.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("clear", 0) >= 0.99 and r.get("unusable", 1) <= 0.01:
                rows.append(r)
    out = []
    for r in rows:
        c, pid, w = r.get("country"), r.get("id"), r.get("window")
        if not (c and pid and w):
            continue
        sr = args.planet_root / c / f"window_{w}" / f"{pid}.tif"
        lbl = args.planet_root / c / "labels" / f"{pid}.tif"
        if sr.exists() and lbl.exists():
            out.append((c, pid, w))
    return out


def _field_frac(planet_root, c, pid):
    p = planet_root / c / "labels" / f"{pid}.tif"
    with rasterio.open(p) as src:
        lbl = src.read(1)
    return float((lbl == 1).sum()) / lbl.size


def _filter_dense(cands, planet_root, min_field_pct, max_check, seed):
    rng = random.Random(seed)
    rng.shuffle(cands)
    kept = []
    for i, (c, pid, w) in enumerate(cands):
        if i >= max_check:
            break
        try:
            frac = _field_frac(planet_root, c, pid)
        except Exception:
            continue
        if frac >= min_field_pct:
            kept.append((frac, (c, pid, w)))
    kept.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in kept]


def _pick_diverse(cands, n, seed):
    by_country = {}
    for c, pid, w in cands:
        by_country.setdefault(c, []).append((c, pid, w))
    rng = random.Random(seed)
    for cs in by_country.values():
        rng.shuffle(cs)
    picks = []
    while len(picks) < n and by_country:
        for c in list(by_country):
            if not by_country[c]:
                del by_country[c]
                continue
            picks.append(by_country[c].pop())
            if len(picks) >= n:
                break
    return picks[:n]


def main():
    args = parse_args()
    cands = _candidates(args)
    print(f"{len(cands)} candidates with clear>=0.99")
    dense = _filter_dense(cands, args.planet_root, args.min_field_pct, args.max_check, args.seed)
    picks = _pick_diverse(dense, args.n, args.seed)
    print(f"selected {len(picks)} patches:")
    for c, pid, w in picks:
        print(f"  {c:14s} {pid}_{w}")

    pairs_per_row = args.pairs_per_row
    n_rows = (len(picks) + pairs_per_row - 1) // pairs_per_row
    _fig, axes = plt.subplots(
        n_rows,
        pairs_per_row * 2,
        figsize=(pairs_per_row * 2 * 1.4, n_rows * 1.55),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r in range(n_rows):
        for k in range(pairs_per_row):
            idx = r * pairs_per_row + k
            base = k * 2
            for c in range(2):
                ax = axes[r, base + c]
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.3)
                    s.set_color("#333333")
            if idx >= len(picks):
                axes[r, base].axis("off")
                axes[r, base + 1].axis("off")
                continue
            country, pid, w = picks[idx]
            sr = args.planet_root / country / f"window_{w}" / f"{pid}.tif"
            lbl = args.planet_root / country / "labels" / f"{pid}.tif"
            rgb = _load_planet_rgb(sr, SQUARE_PX)
            label = _load_label(lbl, SQUARE_PX)
            ov = _overlay(rgb, label)

            axes[r, base + 0].imshow(rgb)
            axes[r, base + 1].imshow(ov)
            if r == 0 and k == 0:
                axes[r, base + 0].set_title("Planet (3 m)", fontsize=8, pad=2)
                axes[r, base + 1].set_title("+ label overlay", fontsize=8, pad=2)
            # Country/patch labels crowd the page-1 hero at camera-ready width.

    plt.tight_layout(pad=0.2, h_pad=0.25, w_pad=0.05)
    plt.savefig(args.out, bbox_inches="tight", dpi=140)
    plt.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
