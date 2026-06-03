"""Measure GPU decompression throughput (nvCOMP) for Deflate vs GDeflate
on real Planet tile data — the whole point of choosing a GPU-decodable
codec over ZSTD.

Reports patches/s of pure GPU decode (one 8-band 512^2 tile = one patch),
compared to the 1,426 patch/s GPU compute ceiling and the ~459 patch/s
CPU-decode path. If GPU decode >> compute ceiling, the codec choice makes
inference compute-bound (the goal) instead of decode-bound.

Uses nvCOMP's native bitstream (encode->decode round-trip) so it measures
GPU decode speed without GDAL/zlib framing concerns; correctness verified.
"""

import argparse
import glob
import time

import cupy as cp
import numpy as np
import rasterio
from nvidia import nvcomp

TILE = 512
COMPUTE_CEIL = 1426.0
CPU_PATH = 459.0


def real_tiles(n: int) -> np.ndarray:
    ap = sorted(glob.glob("data/planet/*/window_a/*.tif"))
    out = []
    for a in ap:
        b = a.replace("/window_a/", "/window_b/")
        with rasterio.open(a) as s:
            A = s.read()
        with rasterio.open(b) as s:
            B = s.read()
        h = min(A.shape[1], B.shape[1], TILE)
        w = min(A.shape[2], B.shape[2], TILE)
        t = np.zeros((8, TILE, TILE), "uint16")
        t[:4, :h, :w] = A[:, :h, :w]
        t[4:, :h, :w] = B[:, :h, :w]
        out.append(t)
        if len(out) >= n:
            break
    return np.stack(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    print(f"nvcomp {nvcomp.__version__}; device {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    tiles = real_tiles(args.tiles)
    print(f"{tiles.shape[0]} real 8-band tiles, {tiles.nbytes / 1e6:.0f} MB raw")
    # each tile -> one nvcomp Array (device), uint8 view of the bytes
    dev = [nvcomp.as_array(cp.asarray(t).view(cp.uint8).ravel()) for t in tiles]
    raw_bytes = tiles.nbytes

    print(f"\n{'codec':10s}{'ratio':>8}{'decode GB/s':>14}{'patch/s':>12}{'vs ceiling':>12}{'vs CPU':>9}")
    for algo in ["Deflate", "GDeflate"]:
        codec = nvcomp.Codec(algorithm=algo)
        comp = codec.encode(dev)
        comp_bytes = sum(c.buffer_size for c in comp)
        # verify round-trip on first tile
        back = codec.decode(comp[:1])
        ok = bool(
            cp.array_equal(
                cp.asarray(back[0]).view(cp.uint16),
                cp.asarray(dev[0]).view(cp.uint16),
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        # warmup
        codec.decode(comp)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            codec.decode(comp)
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        pps = tiles.shape[0] * args.iters / dt
        gbs = raw_bytes * args.iters / dt / 1e9
        print(
            f"{algo:10s}{raw_bytes / comp_bytes:7.2f}x{gbs:13.1f}{pps:12,.0f}"
            f"{pps / COMPUTE_CEIL:11.1f}x{pps / CPU_PATH:8.1f}x"
            f"{'' if ok else '  ROUNDTRIP-FAIL'}"
        )

    print(f"\nReference: GPU compute ceiling {COMPUTE_CEIL:,.0f} p/s; CPU-decode path {CPU_PATH:,.0f} p/s.")
    print("If decode p/s >> ceiling -> GPU decode is free, inference is compute-bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
