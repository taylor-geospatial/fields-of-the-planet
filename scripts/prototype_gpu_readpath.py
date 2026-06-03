"""Prototype the REAL GPU read path for DEFLATE+predictor GeoTIFFs.

Pipeline (no CPU decode, no GDAL in the hot loop):
  1. parse TIFF directory (tifffile) -> per-tile file offsets + byte counts
  2. read compressed tile blobs; strip the 2-byte zlib header + 4-byte adler
     trailer that libtiff/GDAL DEFLATE wraps each tile in -> raw DEFLATE
  3. nvCOMP batched RAW Deflate decode on GPU
  4. undo PREDICTOR=2 (horizontal differencing) on GPU via cumsum
  5. reassemble the (C,H,W) scene on GPU, zero-copy to torch (DLPack)
  6. unfold overlap tiles, bf16 inference

Correctness is verified tile-for-tile against rasterio before timing (hard
assert — no silent pass). Reports end-to-end patch/s vs the 1,426 compute
ceiling and the ~459 CPU-decode path.

Build a test scene first with build_scene.py (writes a deflate_pred/ dir).
"""

import argparse
import glob
import time

import cupy as cp
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import tifffile
import torch
from nvidia import nvcomp

TILE = 512
COMPUTE_CEIL = 1426.0
CPU_PATH = 459.0


def parse_tiff(path: str) -> dict:
    with tifffile.TiffFile(path) as tf:
        p = tf.pages[0]
        assert p.compression == 8 or p.compression == 32946, f"not deflate: {p.compression}"
        assert int(p.predictor) == 2, f"need horizontal predictor, got {p.predictor}"
        assert p.planarconfig == 1, "need contig (pixel-interleaved) planarconfig"
        assert p.dtype == np.uint16
        return dict(
            offsets=list(p.dataoffsets),
            counts=list(p.databytecounts),
            tw=p.tilewidth,
            tl=p.tilelength,
            H=p.imagelength,
            W=p.imagewidth,
            C=p.samplesperpixel,
        )


def read_blobs(path: str, meta: dict) -> list[bytes]:
    blobs = []
    with open(path, "rb") as f:
        for off, cnt in zip(meta["offsets"], meta["counts"]):
            f.seek(off)
            blobs.append(f.read(cnt)[2:-4])  # strip zlib header(2)+adler32(4) -> raw deflate
    return blobs


def decode_scene(blobs: list[bytes], meta: dict, codec: nvcomp.Codec) -> cp.ndarray:
    """blobs -> (C,H,W) uint16 cupy array, predictor undone, on GPU."""
    # one batched host concat + single H2D, then slice views per tile (no per-tile copy)
    lengths = np.fromiter((len(b) for b in blobs), dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(lengths)])
    dbuf = cp.asarray(np.frombuffer(b"".join(blobs), dtype=np.uint8))
    srcs = [nvcomp.as_array(dbuf[offs[i] : offs[i + 1]]) for i in range(len(blobs))]
    out = codec.decode(srcs)
    tl, tw, C = meta["tl"], meta["tw"], meta["C"]
    stacked = cp.stack([cp.asarray(o).view(cp.uint8) for o in out]).view(cp.uint16).reshape(
        -1, tl, tw, C
    )
    # undo horizontal predictor: cumsum along width. uint16 cumsum wraps mod 2^16,
    # which IS the predictor reconstruction -> no wide temporary needed.
    stacked = cp.cumsum(stacked, axis=2, dtype=cp.uint16)
    gh, gw = meta["H"] // tl, meta["W"] // tw
    grid = stacked.reshape(gh, gw, tl, tw, C)
    return grid.transpose(4, 0, 2, 1, 3).reshape(C, gh * tl, gw * tw)  # (C,H,W)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True, help="dir of deflate+predictor scenes")
    p.add_argument("--overlap", type=int, default=102)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=8)
    args = p.parse_args()

    scenes = sorted(glob.glob(f"{args.scene_dir}/*.tif"))
    device = torch.device("cuda:0")
    codec = nvcomp.Codec(algorithm="Deflate", bitstream_kind=nvcomp.BitstreamKind.RAW)
    stride = TILE - args.overlap
    print(f"nvcomp {nvcomp.__version__}; {len(scenes)} deflate+pred scenes; stride {stride}")

    # ---- correctness: GPU-decoded scene must equal rasterio's decode ----
    meta0 = parse_tiff(scenes[0])
    gpu = decode_scene(read_blobs(scenes[0], meta0), meta0, codec)
    with rasterio.open(scenes[0]) as s:
        ref = s.read()  # (C,H,W) uint16
    match = np.array_equal(cp.asnumpy(gpu), ref)
    print(f"correctness vs rasterio: {'PASS' if match else 'FAIL'}")
    if not match:
        raise SystemExit("GPU decode does not match rasterio — aborting")

    model = (
        smp.Unet(encoder_name="efficientnet-b3", in_channels=9, classes=3, encoder_weights=None)
        .eval()
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    metas = [parse_tiff(s) for s in scenes]
    blobs_all = [read_blobs(s, m) for s, m in zip(scenes, metas)]  # host blobs (simulate NVMe-resident)

    def infer_scene(scene: cp.ndarray) -> int:
        x = torch.from_dlpack(scene).to(torch.float32) / 10000.0
        H, W = x.shape[1], x.shape[2]
        gsd = torch.full((1, H, W), 0.30, device=device)
        x = torch.cat([x, gsd], 0)
        t = x.unfold(1, TILE, stride).unfold(2, TILE, stride)
        t = t.permute(1, 2, 0, 3, 4).reshape(-1, 9, TILE, TILE).contiguous(
            memory_format=torch.channels_last
        )
        n = 0
        for i in range(0, t.shape[0], args.batch):
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                model(t[i : i + args.batch]).argmax(1)
            n += min(args.batch, t.shape[0] - i)
        return n

    # warmup
    infer_scene(decode_scene(blobs_all[0], metas[0], codec))
    torch.cuda.synchronize()

    dec_t = inf_t = total = 0
    for _ in range(args.iters):
        for blobs, meta in zip(blobs_all, metas):
            torch.cuda.synchronize(); a = time.perf_counter()
            scene = decode_scene(blobs, meta, codec)
            torch.cuda.synchronize(); b = time.perf_counter()
            total += infer_scene(scene)
            torch.cuda.synchronize(); c = time.perf_counter()
            dec_t += b - a
            inf_t += c - b
    e2e = total / (dec_t + inf_t)
    print(f"\nstage breakdown:  decode+assemble {total / dec_t:,.0f} p/s   infer {total / inf_t:,.0f} p/s")
    print(f"END-TO-END GPU read path: {e2e:,.0f} patch/s")
    print(f"  = {e2e / COMPUTE_CEIL:.0%} of compute ceiling ({COMPUTE_CEIL:,.0f})")
    print(f"  = {e2e / CPU_PATH:.1f}x the CPU-decode path ({CPU_PATH:,.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
