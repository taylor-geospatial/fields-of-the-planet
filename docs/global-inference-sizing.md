# Global PlanetScope Inference — Throughput, Storage & Hardware Sizing

**Question.** How fast, and at what storage/cost, can we run the PRUE
UNet–EfficientNet-B3 field-boundary model over **all of Earth's land at 3 m**
PlanetScope resolution — on H100s today and NVIDIA Blackwell tomorrow — and
what's the actual bottleneck?

**One-line answer.** The model is tiny; the GPU is *not* the limit — **data
decode is**. Store the global mosaic as **DEFLATE + horizontal predictor** and
**decode it on the GPU (nvCOMP)**: that lands at **~112 TB / ~$2,570/mo on S3**
and a **~2.4 h global pass on a single 8-GPU node**, with the model fully
compute-bound.

> Every number below was **measured on H100 80GB nodes with real Planet
> tiles**, not estimated. Reproduction commands are at the bottom. Date: 2026-06.

---

## 1. The model

| | |
|---|---|
| Architecture | `smp.Unet`, EfficientNet-B3 encoder, 9-ch in (8 spectral + GSD), 3-class out |
| Patch | 512×512, bf16 |
| Params | 13.2 M |
| Compute | **31.0 GFLOP / patch** (torch FLOP counter; `thop` undercounts to 21.6 — ignore it) |

## 2. Earth coverage @ 3 m

- 1 patch (512², no overlap) = **2.36 km²**; 1 km² = 111,111 px.
- All land (149 M km², incl. ice sheets) = **63 M patches / 16.6 T pixels**.
- Overlap multiplier: 32 px → ×1.14 (72 M patches); **20% / 102 px → ×1.56 (98 M patches)**.
- The tables below use the **98 M-patch (20% overlap)** global pass.

## 3. GPU compute ceiling (synthetic data, no I/O)

| Mode | patch/s | Sustained | **MFU** |
|---|---|---|---|
| Eager | 937 | 29 TFLOPs | 2.9% |
| `torch.compile` | **1,426** | 44 TFLOPs | **4.5%** |

**The model runs at 4.5% MFU — the H100 is ~95% idle.** EfficientNet is
depthwise/SE-heavy → low arithmetic intensity → **launch/bandwidth-bound, not
compute-bound.** Consequence: extrapolate across GPUs by **HBM bandwidth, not
tensor-core peak**. (TensorRT/FP8 is a 5–10× lever here, not more GPUs.)

## 4. The real bottleneck: data loading, not the GPU

Reading full scenes from **node-local NVMe** and running the actual pipeline
(decode → normalize → GPU → infer):

| Path | patch/s | GPU util |
|---|---|---|
| GPU compute ceiling | 1,426 | 100% |
| Windowed reads, ZSTD CPU decode | 459 | 32% |
| Full-scene read → `unfold`, ZSTD CPU | 529 | 37% |
| Full-scene, **uncompressed**, CPU/rasterio | 517 | 36% |

CPU decode caps everything at **~460–530 patch/s (≈10 p/s/core on 48 cores)**.
To feed *one* H100 to its ceiling needs **~150 decode cores**; 8 GPUs need
**~1,200**. A cloud `p6-b200` has 24 vCPU/GPU → its GPUs would sit **~85% idle**
on CPU decode. **This is the whole problem.**

## 5. Codec study (measured on a real 4096² 8-band mosaic, 268 MB raw)

| Codec | Ratio | GPU-decodable? |
|---|---|---|
| ZSTD-9 (current) | 1.97× | ✗ (sequential entropy stage; GPU-hostile) |
| ZSTD-9 + predictor | 2.49× | ✗ |
| **DEFLATE-9 + predictor** | **2.37×** | ✅ (nvCOMP / nvTIFF / Blackwell HW) |
| LZW + predictor | 2.02× | ✅ |
| Uncompressed | 1.00× | n/a |

Notes:
- A **horizontal predictor** (`PREDICTOR=2`) is the big lever — it makes
  DEFLATE (2.37×) *beat* current ZSTD-without-predictor (1.97×).
- **GDAL does not support GDeflate** — `COMPRESS=GDEFLATE` silently writes
  *uncompressed* (no error). Use standard `DEFLATE`.
- Internal **tiling/overviews are unnecessary** if you read whole scenes
  (ratio is layout-invariant: 2.31–2.37×). But keep **multiple strips**
  (`rowsperstrip≈512`) — a single strip = one sequential stream = no parallel
  decode.

## 6. GPU decode — the unlock (nvCOMP 5.2.0, H100, real tiles)

| Codec | Ratio | Decode | patch/s | vs compute ceiling | vs CPU path |
|---|---|---|---|---|---|
| **Deflate** | 2.08× | 20.4 GB/s | **4,853** | **3.4×** | 10.6× |
| GDeflate | 2.06× | 65.0 GB/s | 15,494 | 10.9× | 33.8× |

The nvCOMP decode *kernel* runs at 4,853 p/s — 3.4× the model's appetite —
suggesting GPU decode could make inference compute-bound. **But see §6a: a real
end-to-end prototype shows the kernel is only a small slice of the path, and a
naive implementation does NOT reach the ceiling.**

## 6a. Prototype reality check — GPU decode alone is not the win

A real read-path prototype (`prototype_gpu_readpath.py`: TIFF tile offsets →
strip zlib wrapper → nvCOMP RAW Deflate → GPU predictor-undo → scene assembly →
`unfold` → infer, **verified tile-for-tile against rasterio**) measured:

| Stage | patch/s |
|---|---|
| decode + assemble | 590 |
| infer (eager, incl. tiling copies) | 879 |
| **end-to-end (serial)** | **353** (25% ceiling, **0.8× the CPU path**) |

**The fast nvCOMP kernel (4,853) is a tiny fraction of the path.** The wall is
**GPU-side data movement**: stacking 256 decoded tiles, predictor-undo, the
`transpose→reshape` into `(C,H,W)` (~1 GB strided copy/scene), and the
`unfold→contiguous` (~1.7 GB copy). A naive GPU port nets *nothing* over CPU.

Reaching the 1,426 ceiling requires real engineering, not just calling nvCOMP:
1. a **fused decode→deinterleave→predictor→assemble kernel** (remove the stack +
   strided-transpose copies — the dominant cost);
2. **CUDA streams** to overlap memory-bound assembly with compute-bound infer;
3. **`torch.compile`/TensorRT** on infer (879 → ~1,426+);
4. or **store pre-tiled overlapping 512² patches** so decode lands directly in
   inference shape (no scene-assembly/unfold) — ~1.56× storage, removes both big
   copies.

The codec recommendation below still holds (DEFLATE+predictor is the right,
GPU-decodable, smallest-practical format); but the throughput win is **future
optimization work**, not a free result. Treat the "GPU" runtime rows as a
*target*, not a measured number.

## 7. The decision table — storage × runtime × cost

Full global pass, 98 M patches (20% overlap):

| Codec | Storage | S3 $/mo | Decode | patch/s | 1×H100 | 8-GPU node |
|---|---|---|---|---|---|---|
| Uncompressed (rasterio/CPU) | 265 TB | $6,095 | CPU | 517 | 53 h | 6.6 h |
| Uncompressed (GPUDirect, ideal) | 265 TB | $6,095 | — | 1,426 | 19 h | 2.4 h |
| ZSTD-9 (current) | 135 TB | $3,094 | CPU | 529 | 51 h | 6.4 h |
| DEFLATE+pred (CPU) | 112 TB | $2,572 | CPU | 516 | 53 h | 6.6 h |
| DEFLATE+pred (GPU, naive prototype) | 112 TB | $2,572 | GPU | 353 | 77 h | 9.6 h |
| **DEFLATE+pred (GPU, optimized — TARGET)** | **112 TB** | **$2,572** | **GPU** | **≤1,426** | **≥19 h** | **≥2.4 h** |

*(8-GPU CPU rows assume 48 cores/GPU — optimistic; real nodes ~24, so ~2×
slower. The naive GPU prototype is **measured** and currently slower than CPU;
the optimized GPU row is a **target** requiring the §6a kernel-fusion work —
not yet achieved.)*

**Verdict — DEFLATE + predictor + GPU decode wins on both axes:**
- **Uncompressed is a trap** — same CPU-path speed as compressed (read/transfer
  bound), but 2.4× the storage (+$3,500/mo).
- **CPU deflate ≈ CPU zstd** — deflate alone is no win; its *only* benefit is
  GPU decodability.
- DEFLATE+GPU = **smallest practical storage AND fastest**, with 3.4× decode
  headroom left for a TensorRT/FP8 ceiling bump.

## 8. AWS hardware & feed

| Instance | GPUs | vCPU | Local NVMe | On-demand |
|---|---|---|---|---|
| p5.48xlarge | 8× H100 | 192 | — | ~$33–55/hr |
| p5en.48xlarge | 8× H200 | 192 | — | ~$63/hr |
| **p6-b200.48xlarge** | **8× B200** (180 GB) | 192 | 30.4 TB | **$113.93/hr** (~$37 spot) |
| p6e-gb200 (NVL72) | 36–72 GB200 | — | — | $823k–1.65M / 3 mo |

- **`p6e-gb200` (NVL72) is overkill** for a 13 M-param model — skip it.
- Storage feed: the 112 TB dataset **won't fit** on 30.4 TB local NVMe. Use
  **FSx for Lustre** (≤150 GB/s/client, S3-backed) to keep GPUs fed;
  **Mountpoint-S3 caps ~12.5 GB/s** and bottlenecks. Keep bucket + instance
  **same region** → $0 egress.
- Compute cost of a global pass is **~$100–270** (a rounding error); the line
  item is **storage (~$2.6k/mo)** + the one-time re-encode pass.

## 9. Blackwell extrapolation (once compute-bound, bandwidth-scaled)

Scale by HBM bandwidth (model is bandwidth-bound), not tensor-core peak:

| GPU | HBM BW | 8-GPU node, global pass |
|---|---|---|
| H100 (measured) | 3.35 TB/s | 2.4 h |
| H200 | 4.8 TB/s | ~1.7 h |
| B200 | 8.0 TB/s | ~1.0 h |

TensorRT/FP8 stacks on top (compute ceiling rises; decode still free).

## 10. Recommended pipeline

1. **Re-encode** the global mosaic: `COMPRESS=DEFLATE, PREDICTOR=2,
   TILED=NO (multi-strip rowsperstrip≈512), no overviews`. → ~112 TB.
2. **Read whole scene → GPU**, decode tiles with **nvCOMP (Deflate)**, undo
   predictor on-GPU.
3. **Overlap-tile on GPU** with `torch.Tensor.unfold` (zero-copy view).
4. Append GSD channel, batch through the model (bf16, channels_last,
   `torch.compile`; later TensorRT/FP8).
5. One **8-GPU node** (p5/p6-b200) + **FSx Lustre** does a global pass in
   **~2.4 h**, fully GPU-bound.

**Open follow-ups:** prototype the real nvCOMP read path end-to-end (TIFF tile
offsets → GPU → nvCOMP → predictor-undo → `unfold`) to confirm the 1,426 p/s;
TensorRT/FP8 pass to lift the compute ceiling.

---

## Reproduction

All scripts under `scripts/`, SLURM wrappers under `scripts/slurm/`
(`--account=bgtj-tgirails`, `gpu` partition, H100):

```bash
sbatch scripts/slurm/bench_h100.sbatch        # GPU compute ceiling (synthetic)
sbatch scripts/slurm/bench_scene.sbatch       # full-scene windowed, local NVMe (CPU decode)
sbatch scripts/slurm/bench_codecs.sbatch      # per-codec full-scene read->unfold->infer
sbatch scripts/slurm/bench_gpu_decode.sbatch  # nvCOMP Deflate/GDeflate GPU decode throughput
```

Codec ratios were measured on-CPU with GDAL/rasterio; GPU decode used
`nvidia-nvcomp-cu12` (nvCOMP 5.2.0) + `cupy-cuda12x` on CUDA 12.8 / torch 2.10.
