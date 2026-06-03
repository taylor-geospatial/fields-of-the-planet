"""End-to-end inference throughput: REAL data loading + decode + GPU.

Unlike bench_h100.py (synthetic on-GPU tensors = compute ceiling), this
exercises the production path: rasterio COG read of two 4-band zstd tiles
per patch, float cast, /10000 norm, pad to 512, GSD channel, host->device,
bf16 inference. Sweeps DataLoader num_workers to find where storage+CPU
decode saturates relative to the GPU.

Reports loader-only patch/s (GPU idle) and end-to-end patch/s per worker
count, so the bottleneck is explicit. Storage here is whatever backs
data/ (NFS on RAILS) — note that when reading the result.
"""

import argparse
import glob
import time

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PLANET_SR_SCALE = 10000.0
TILE = 512


def collect_pairs(root: str, limit: int) -> list[tuple[str, str]]:
    pairs = []
    for a in sorted(glob.glob(f"{root}/*/window_a/*.tif")):
        b = a.replace("/window_a/", "/window_b/")
        pairs.append((a, b))
        if len(pairs) >= limit:
            break
    return pairs


class PatchSet(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> torch.Tensor:
        a_path, b_path = self.pairs[i]
        with rasterio.open(a_path) as s:
            a = s.read().astype(np.float32)  # (4,H,W)
        with rasterio.open(b_path) as s:
            b = s.read().astype(np.float32)  # (4,H,W)
        h = min(a.shape[1], b.shape[1])
        w = min(a.shape[2], b.shape[2])
        x = np.concatenate([a[:, :h, :w], b[:, :h, :w]], axis=0) / PLANET_SR_SCALE  # (8,h,w)
        t = torch.from_numpy(x)
        # pad/crop to TILE
        ph, pw = max(0, TILE - t.shape[1]), max(0, TILE - t.shape[2])
        t = F.pad(t, (0, pw, 0, ph))[:, :TILE, :TILE]
        gsd = torch.full((1, TILE, TILE), 0.30)  # Planet GSD channel (3/10)
        return torch.cat([t, gsd], dim=0)  # (9,512,512)


def run(dl: DataLoader, model, device, gpu: bool, max_patches: int) -> float:
    n = 0
    # warmup one batch (worker spin-up excluded)
    it = iter(dl)
    first = next(it)
    if gpu:
        x = first.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            model(x).argmax(1)
        torch.cuda.synchronize()
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
    p.add_argument("--root", default="data/planet")
    p.add_argument("--workers", type=int, nargs="*", default=[8, 16, 32, 48])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-patches", type=int, default=4000)
    p.add_argument("--pool", type=int, default=6000)
    args = p.parse_args()

    pairs = collect_pairs(args.root, args.pool)
    print(f"collected {len(pairs)} patch pairs from {args.root}")
    ds = PatchSet(pairs)

    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(0)}")
    model = (
        smp.Unet(encoder_name="efficientnet-b3", in_channels=9, classes=3, encoder_weights=None)
        .eval()
        .to(device)
        .to(memory_format=torch.channels_last)
    )

    print(f"\n{'workers':>8}{'loader-only p/s':>18}{'end-to-end p/s':>17}{'GPU util*':>11}")
    GPU_CEIL = 1426.0  # measured compute ceiling, 1 H100
    for nw in args.workers:
        common = dict(
            batch_size=args.batch,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        dl_load = DataLoader(ds, shuffle=False, **common)
        load_ps = run(dl_load, model, device, gpu=False, max_patches=args.max_patches)
        del dl_load
        dl_e2e = DataLoader(ds, shuffle=False, **common)
        e2e_ps = run(dl_e2e, model, device, gpu=True, max_patches=args.max_patches)
        del dl_e2e
        print(f"{nw:>8}{load_ps:>18,.0f}{e2e_ps:>17,.0f}{e2e_ps / GPU_CEIL:>10.0%}")

    print("\n*GPU util = end-to-end / measured compute ceiling (1426 p/s on H100).")
    print("Storage = NFS (data/ mount). Local NVMe / FSx would raise loader-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
