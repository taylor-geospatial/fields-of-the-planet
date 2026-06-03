"""Full-scene inference throughput per codec: read whole scene -> GPU ->
unfold overlap tiles -> infer. Matches the planned production pipeline
(no windowed/partial-block reads).

Scenes decode concurrently in reader threads (GDAL releases the GIL during
read, so threads parallelize decode), overlapping CPU decode with GPU
compute. Reports end-to-end patch/s for the given scene directory's codec.
"""

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("GDAL_NUM_THREADS", "1")  # parallelism is across scenes, not within

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch

SCALE = 10000.0
TILE = 512


def load_scene(path: str) -> np.ndarray:
    with rasterio.open(path) as s:
        return s.read().astype(np.float32)  # (8,H,W)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--overlap", type=int, default=102)
    p.add_argument("--readers", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    scenes = sorted(glob.glob(f"{args.scene_dir}/*.tif"))
    stride = TILE - args.overlap
    device = torch.device("cuda:0")
    model = (
        smp.Unet(encoder_name="efficientnet-b3", in_channels=9, classes=3, encoder_weights=None)
        .eval()
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    def infer_scene(arr: np.ndarray) -> int:
        H, W = arr.shape[1], arr.shape[2]
        x = torch.from_numpy(arr).to(device, non_blocking=True) / SCALE
        gsd = torch.full((1, H, W), 0.30, device=device)
        x = torch.cat([x, gsd], 0)  # (9,H,W)
        t = x.unfold(1, TILE, stride).unfold(2, TILE, stride)  # (9,nh,nw,512,512)
        t = t.permute(1, 2, 0, 3, 4).reshape(-1, 9, TILE, TILE).contiguous(
            memory_format=torch.channels_last
        )
        n = 0
        for i in range(0, t.shape[0], args.batch):
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                model(t[i : i + args.batch]).argmax(1)
            n += min(args.batch, t.shape[0] - i)
        return n

    ex = ThreadPoolExecutor(max_workers=args.readers)
    futures = [ex.submit(load_scene, s) for s in scenes]

    total, t0, warmed = 0, None, False
    for fut in as_completed(futures):
        arr = fut.result()
        if not warmed:  # exclude first scene (worker spin-up + CUDA warmup)
            infer_scene(arr)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            warmed = True
            continue
        total += infer_scene(arr)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    pps = total / dt
    codec = os.path.basename(args.scene_dir.rstrip("/"))
    print(f"  {codec:14s} readers={args.readers}  {pps:8,.0f} patch/s  ({pps / 1426:.0%} of GPU ceiling)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
