"""Assemble real Planet tiles into full-scene COGs, in multiple codecs.

Production inference reads big scenes and tiles them, not millions of tiny
files. This builds representative full scenes on node-local storage from
REAL radiometry (faithful compression ratio + decode cost), writing each
scene simultaneously in several codecs from ONE read pass over the source
so the formats are byte-identical in content and differ only in encoding.

Codecs: zstd (current), deflate+predictor2, uncompressed (none).
"""

import argparse
import glob
import os

import numpy as np
import rasterio
from rasterio.windows import Window

TILE = 512

FORMATS = {
    "zstd": dict(compress="zstd", zstd_level=9),
    "deflate_pred": dict(compress="deflate", zlevel=9, predictor=2),
    "none": dict(compress=None),
}


def pairs(root: str, n: int) -> list[tuple[str, str]]:
    out = []
    for a in sorted(glob.glob(f"{root}/*/window_a/*.tif")):
        out.append((a, a.replace("/window_a/", "/window_b/")))
        if len(out) >= n:
            break
    return out


def read_8band_512(a_path: str, b_path: str) -> np.ndarray:
    with rasterio.open(a_path) as s:
        a = s.read()
    with rasterio.open(b_path) as s:
        b = s.read()
    h = min(a.shape[1], b.shape[1], TILE)
    w = min(a.shape[2], b.shape[2], TILE)
    out = np.zeros((8, TILE, TILE), dtype="uint16")
    out[:4, :h, :w] = a[:, :h, :w]
    out[4:, :h, :w] = b[:, :h, :w]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--src", default="data/planet")
    p.add_argument("--scenes", type=int, default=6)
    p.add_argument("--size", type=int, default=8192)
    args = p.parse_args()

    blocks = args.size // TILE
    per_scene = blocks * blocks
    src = pairs(args.src, args.scenes * per_scene)
    print(f"need {args.scenes * per_scene} patches, have {len(src)}")
    for fmt in FORMATS:
        os.makedirs(f"{args.out_dir}/{fmt}", exist_ok=True)

    base = dict(
        driver="GTiff",
        height=args.size,
        width=args.size,
        count=8,
        dtype="uint16",
        tiled=True,
        blockxsize=TILE,
        blockysize=TILE,
    )
    raw = args.size * args.size * 8 * 2 / 1e6
    k = 0
    for s in range(args.scenes):
        writers = {
            fmt: rasterio.open(f"{args.out_dir}/{fmt}/scene_{s:02d}.tif", "w", **base, **opt)
            for fmt, opt in FORMATS.items()
        }
        for bi in range(blocks):
            for bj in range(blocks):
                a, b = src[k % len(src)]
                k += 1
                tile = read_8band_512(a, b)
                win = Window(bj * TILE, bi * TILE, TILE, TILE)
                for w in writers.values():
                    w.write(tile, window=win)
        for fmt, w in writers.items():
            w.close()
            mb = os.path.getsize(f"{args.out_dir}/{fmt}/scene_{s:02d}.tif") / 1e6
            print(f"  scene_{s:02d} {fmt:13s} {mb:6.0f} MB ({raw / mb:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
