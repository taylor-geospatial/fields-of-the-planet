"""Optimized GPU read path: PRE-TILED storage.

The naive scene path is bottlenecked by GPU data movement (scene assembly +
transpose + unfold copies), not by nvCOMP decode. This restructures it: store
overlapping 512^2 patches, each DEFLATE+predictor compressed, so nvCOMP decodes
straight into inference shape (NHWC uint16 = torch channels_last) with NO scene
assembly, NO unfold, NO strided transpose. GSD is baked in as a 9th band.

Predictor-undo stays a cheap in-tile cumsum. Per-batch float cast only.
Measures decode+assemble vs infer (eager and torch.compile) vs end-to-end.

Patches are built in-memory from real tiles (predictor-diff + raw-deflate) to
isolate the GPU pipeline; on-disk reads of pre-tiled patches are trivial
sequential NVMe.
"""

import argparse
import glob
import time
import zlib

import cupy as cp
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from nvidia import nvcomp

TILE = 512
GSD = 3000  # 0.30 * 10000
COMPUTE_CEIL = 1426.0


def real_patches(n: int) -> np.ndarray:
    """n patches of (TILE,TILE,9) uint16 (NHWC): 8 real bands + constant GSD band."""
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
        t = np.zeros((9, TILE, TILE), "uint16")
        t[:4, :h, :w] = A[:, :h, :w]
        t[4:8, :h, :w] = B[:, :h, :w]
        t[8] = GSD
        out.append(np.ascontiguousarray(t.transpose(1, 2, 0)))  # -> (H,W,C)
        if len(out) >= n:
            break
    return np.stack(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda:0")
    print(f"nvcomp {nvcomp.__version__}; {torch.cuda.get_device_name(0)}")
    patches = real_patches(args.tiles)  # (N,H,W,C) uint16
    N = patches.shape[0]
    print(f"{N} pre-tiled patches (NHWC, 9-band)")

    codec = nvcomp.Codec(algorithm="Deflate", bitstream_kind=nvcomp.BitstreamKind.RAW)

    def build_grouped(G: int):
        """Group G patches per deflate block -> N/G large contiguous blocks
        (fewer nvCOMP outputs to gather)."""
        diffed = patches.copy()
        diffed[:, :, 1:, :] = (patches[:, :, 1:, :] - patches[:, :, :-1, :]).astype(np.uint16)
        blobs = []
        for g0 in range(0, N, G):
            co = zlib.compressobj(6, zlib.DEFLATED, -15)
            blobs.append(co.compress(diffed[g0 : g0 + G].tobytes()) + co.flush())
        lengths = np.fromiter((len(b) for b in blobs), dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(lengths)])
        host = np.frombuffer(b"".join(blobs), dtype=np.uint8)
        return host, offs, len(blobs), sum(lengths)

    def make_decode(G: int):
        host, offs, ngrp, _ = build_grouped(G)

        def decode() -> cp.ndarray:
            dbuf = cp.asarray(host)
            srcs = [nvcomp.as_array(dbuf[offs[i] : offs[i + 1]]) for i in range(ngrp)]
            out = codec.decode(srcs)
            arr = cp.concatenate([cp.asarray(o).view(cp.uint16) for o in out]).reshape(
                N, TILE, TILE, 9
            )
            return cp.cumsum(arr, axis=2, dtype=cp.uint16)  # undo predictor, NHWC

        return decode

    decode = make_decode(1)
    dec = cp.asnumpy(decode())
    print(f"correctness vs original: {'PASS' if np.array_equal(dec, patches) else 'FAIL'}")
    if not np.array_equal(dec, patches):
        raise SystemExit("decode mismatch")

    base = smp.Unet(encoder_name="efficientnet-b3", in_channels=9, classes=3, encoder_weights=None)
    base = base.eval().to(device).to(memory_format=torch.channels_last)

    # uncompressed pre-tiled "load" = just H2D the raw patches (no decode/assemble)
    host_raw = cp.cuda.alloc_pinned_memory(patches.nbytes)
    pinned = np.frombuffer(host_raw, dtype=np.uint16, count=patches.size).reshape(patches.shape)
    pinned[:] = patches

    def load_uncompressed() -> cp.ndarray:
        return cp.asarray(pinned)  # single H2D, already (N,512,512,9) NHWC

    def make_infer(model):
        def infer(scene: cp.ndarray) -> int:
            x = torch.from_dlpack(scene).permute(0, 3, 1, 2)  # channels_last view, no copy
            n = 0
            for i in range(0, N, args.batch):
                b = x[i : i + args.batch].to(torch.float32) / 10000.0
                with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                    model(b).argmax(1)
                n += b.shape[0]
            return n
        return infer

    def measure(tag: str, load, infer) -> None:
        infer(load())  # warmup (compile)
        torch.cuda.synchronize()
        load_t = inf_t = total = 0
        for _ in range(args.iters):
            torch.cuda.synchronize(); a = time.perf_counter()
            scene = load()
            torch.cuda.synchronize(); b = time.perf_counter()
            total += infer(scene)
            torch.cuda.synchronize(); c = time.perf_counter()
            load_t += b - a
            inf_t += c - b
        e2e = total / (load_t + inf_t)
        print(
            f"[{tag:24s}] load {total / load_t:6,.0f}  infer {total / inf_t:6,.0f}  "
            f"end-to-end {e2e:6,.0f} p/s ({e2e / COMPUTE_CEIL:.0%} ceiling)"
        )

    compiled = torch.compile(base)
    cinfer = make_infer(compiled)
    for G in (1, 32, 128):
        measure(f"deflate group={G:<3d} compiled", make_decode(G), cinfer)
    measure("uncompressed compiled", load_uncompressed, cinfer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
