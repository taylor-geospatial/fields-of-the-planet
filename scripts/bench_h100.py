"""Pure GPU-compute throughput benchmark for the PRUE UNet-efnet-b3 model.

Isolates the GPU ceiling: synthetic data is generated ON the GPU, so no
dataloader / NVMe / decode in the loop. Reports patches/s, pixels/s,
sustained TFLOPs and MFU at the largest batch that fits, in bf16.

Throughput is weight-independent, so we build the architecture fresh
(no checkpoint needed) — identical to the trained config:
  smp.Unet(efficientnet-b3, in_channels=9, classes=3).

Run via SLURM wrapper scripts/slurm/bench_h100.sbatch.
"""

import argparse
import time

import segmentation_models_pytorch as smp
import torch

FLOP_PER_PATCH = 30.96e9  # measured via torch.utils.flop_counter, 512x512
H100_BF16_PEAK = 989.4e12  # SXM dense, no sparsity


def build_model() -> torch.nn.Module:
    return smp.Unet(
        encoder_name="efficientnet-b3", in_channels=9, classes=3, encoder_weights=None
    )


@torch.inference_mode()
def bench_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch: int,
    size: int,
    channels_last: bool,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    x = torch.randn(batch, 9, size, size, device=device, dtype=torch.float32).to(
        memory_format=mem_fmt
    )
    # warmup
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    pps = batch * iters / dt
    tflops = pps * FLOP_PER_PATCH / 1e12
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    return {
        "batch": batch,
        "patch_per_s": pps,
        "tflops": tflops,
        "mfu": tflops * 1e12 / H100_BF16_PEAK,
        "peak_mem_gb": peak_mem,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batches", type=int, nargs="*", default=[8, 16, 32, 64, 96, 128, 192, 256])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}")
    print(f"FLOP/patch={FLOP_PER_PATCH/1e9:.2f} G  size={args.size}  compile={args.compile}")

    model = build_model().eval().to(device).to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)

    print(f"{'batch':>6}{'patch/s':>12}{'Gpix/s':>10}{'TFLOPs':>9}{'MFU':>7}{'mem GB':>9}")
    best = None
    for b in args.batches:
        try:
            r = bench_batch(model, device, b, args.size, True, args.iters, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f"{b:>6}  OOM")
            torch.cuda.empty_cache()
            continue
        gpix = r["patch_per_s"] * args.size * args.size / 1e9
        print(
            f"{r['batch']:>6}{r['patch_per_s']:>12,.0f}{gpix:>10.2f}"
            f"{r['tflops']:>9.1f}{r['mfu']:>7.1%}{r['peak_mem_gb']:>9.1f}"
        )
        if best is None or r["patch_per_s"] > best["patch_per_s"]:
            best = r
        torch.cuda.reset_peak_memory_stats()

    # Earth extrapolation from the best measured throughput
    patches_global = 149e6 / (args.size * args.size * 9 / 1e6)
    secs = patches_global / best["patch_per_s"]
    print(
        f"\nBEST: {best['patch_per_s']:,.0f} patch/s @ batch {best['batch']} "
        f"({best['mfu']:.1%} MFU)"
    )
    print(
        f"Global land (149M km2 = {patches_global/1e6:.0f}M patches): "
        f"{secs/3600:.2f} h on 1 GPU  |  {secs/3600/8:.2f} h on 8 GPUs"
    )
    print("(GPU-compute floor only; excludes I/O, decode, reproject, mosaicking.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
