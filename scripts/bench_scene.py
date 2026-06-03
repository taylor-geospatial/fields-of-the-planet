"""End-to-end inference throughput reading FULL scenes from local storage.

Production path: open big scene COGs once, windowed-read 512^2 tiles with
overlap stride, decode (zstd block), normalize, GSD channel, bf16 infer.
This is the realistic I/O profile (file-open cost amortized; per-block
zstd decode dominates), reading from node-local NVMe/tmp, not NFS.

Sweeps DataLoader num_workers; reports loader-only vs end-to-end patch/s
so the bottleneck (storage+decode vs GPU) is explicit, then extrapolates
to a global pass.
"""

import argparse
import glob
import os
import time

# Pin GDAL to 1 thread per worker: with N dataloader workers, GDAL's own
# threadpool oversubscribes the node (the cause of 48-worker collapse).
os.environ.setdefault("GDAL_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

PLANET_SR_SCALE = 10000.0
TILE = 512
GPU_CEIL = 1426.0  # measured 1-H100 compute ceiling (compiled), patch/s


def build_windows(scenes: list[str], stride: int) -> list[tuple[str, int, int]]:
    wins = []
    for sc in scenes:
        with rasterio.open(sc) as s:
            H, W = s.height, s.width
        for r in range(0, H - TILE + 1, stride):
            for c in range(0, W - TILE + 1, stride):
                wins.append((sc, r, c))
    return wins


class SceneWindows(Dataset):
    def __init__(self, wins: list[tuple[str, int, int]]):
        self.wins = wins
        self._handles: dict[str, rasterio.DatasetReader] = {}  # per-worker open-handle cache

    def __len__(self) -> int:
        return len(self.wins)

    def _open(self, sc: str) -> rasterio.DatasetReader:
        h = self._handles.get(sc)
        if h is None:
            h = rasterio.open(sc)  # kept open for the worker's lifetime
            self._handles[sc] = h
        return h

    def __getitem__(self, i: int) -> torch.Tensor:
        sc, r, c = self.wins[i]
        s = self._open(sc)
        x = s.read(window=Window(c, r, TILE, TILE)).astype(np.float32)  # (8,512,512)
        x = x / PLANET_SR_SCALE
        t = torch.from_numpy(x)
        gsd = torch.full((1, TILE, TILE), 0.30)
        return torch.cat([t, gsd], dim=0)  # (9,512,512)


def run(dl: DataLoader, model, device, gpu: bool, max_patches: int) -> float:
    it = iter(dl)
    first = next(it)  # warmup / worker spin-up excluded
    if gpu:
        x = first.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            model(x).argmax(1)
        torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    for batch in it:
        if gpu:
            x = batch.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                model(x).argmax(1)
        n += batch.shape[0]
        if n >= max_patches:
            break
    if gpu:
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--overlap", type=int, default=102)  # 20% of 512
    p.add_argument("--workers", type=int, nargs="*", default=[8, 16, 24, 32, 48])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-patches", type=int, default=4000)
    args = p.parse_args()

    scenes = sorted(glob.glob(f"{args.scene_dir}/*.tif"))
    stride = TILE - args.overlap
    wins = build_windows(scenes, stride)
    print(f"{len(scenes)} scenes, stride {stride} (overlap {args.overlap}px) -> {len(wins)} windows")
    ds = SceneWindows(wins)

    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}")
    model = (
        smp.Unet(encoder_name="efficientnet-b3", in_channels=9, classes=3, encoder_weights=None)
        .eval()
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    print(f"\n{'workers':>8}{'loader-only p/s':>18}{'end-to-end p/s':>17}{'GPU util':>10}")
    best = 0.0
    for nw in args.workers:
        common = dict(
            batch_size=args.batch,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        load_ps = run(DataLoader(ds, **common), model, device, False, args.max_patches)
        e2e_ps = run(DataLoader(ds, **common), model, device, True, args.max_patches)
        best = max(best, e2e_ps)
        print(f"{nw:>8}{load_ps:>18,.0f}{e2e_ps:>17,.0f}{e2e_ps / GPU_CEIL:>9.0%}")

    # global extrapolation from best end-to-end, 20% overlap = 98M patches
    patches = 98e6
    secs = patches / best
    print(f"\nBEST end-to-end: {best:,.0f} patch/s on 1 H100 (vs {GPU_CEIL:,.0f} compute ceiling)")
    print(f"Global pass (98M patches, 20% overlap):")
    print(f"  1x H100:  {secs / 3600:5.1f} h   |  8-GPU node: {secs / 3600 / 8:4.1f} h")
    print("Storage: node-local NVMe/tmp.  (compute ceiling shown for reference only.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
