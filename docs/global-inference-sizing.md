# Global PlanetScope Inference — Throughput, Storage & Hardware Sizing

**Question.** How fast, and at what storage/cost, can we run the PRUE
UNet–EfficientNet-B3 field-boundary model over **all of Earth's land at 3 m**
PlanetScope resolution — on H100s today and NVIDIA Blackwell tomorrow — and
what's the actual bottleneck?

**One-line answer.** The model is tiny; the GPU is *not* the limit — **getting
pixels onto it is**. Inference compiles to 95% of the H100 ceiling, but a global
pass is gated by the data path. Measured options: **uncompressed pre-tiled +
`torch.compile` → 1,203 p/s (compute-bound, ~2.9 h/8-GPU node) at ~413 TB**, vs
**DEFLATE+predictor → 112 TB but ~CPU-class throughput** (the off-the-shelf GPU
decode glue is the wall). Compact *and* fast needs a fused decode kernel (§6a).
It's a storage ↔ throughput tradeoff, not a free win.

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

## 6a. GPU read-path prototype & optimization (measured)

A real read-path prototype (`prototype_gpu_readpath.py` / `bench_pretiled.py`:
TIFF tile offsets → strip zlib wrapper → nvCOMP RAW Deflate → GPU predictor-undo
→ tile/scene assembly → infer, **verified tile-for-tile against rasterio**) was
built and progressively optimized. Results on H100:

| Path | load p/s | infer p/s | end-to-end | % ceiling |
|---|---|---|---|---|
| naive scene (decode→assemble→unfold), eager | — | — | 353 | 25% |
| deflate pre-tiled + nvCOMP, compiled | 765 | 1,354 | 489 | 34% |
| deflate pre-tiled, grouped 32 / 128 | 37 / 9 | — | 36 / 9 | worse |
| **uncompressed pre-tiled, compiled** | **10,436** | 1,359 | **1,203** | **84%** |

What was learned:
- **Inference is solved**: `torch.compile` → ~1,350 p/s (95% of the 1,426
  ceiling). Pre-tiling (store overlapping 512² patches → decode lands in NHWC =
  channels_last, no scene-assembly/unfold/strided-transpose) removed the big
  copies that crippled the naive scene path.
- **Uncompressed pre-tiled reaches 1,203 p/s (84% ceiling, 2.6× the CPU path)** —
  load is essentially free (10,436), so it's compute-bound. This is the
  practical fast path, **off-the-shelf, today**.
- **Compressed (deflate) GPU decode does NOT pay off with current tooling.** The
  nvCOMP *kernel* is fast (4,853 p/s) but the **gather of decoded tiles +
  predictor cumsum** caps the path at ~489 (≈ CPU level). **Grouping patches
  into larger deflate blocks makes it far worse** (37 / 9 p/s) — deflate is
  sequential *within* a stream; nvCOMP parallelizes *across many small* streams,
  so big blocks serialize the decode. Closing this needs a **fused custom CUDA
  kernel** (decode→contiguous→predictor in one pass) — a real engineering
  project, not a config.
- **nvTIFF** (purpose-built GPU TIFF decoder) *correctly* reads our 8-band uint16
  format but its **GPU decode FAILS on DEFLATE+predictor** (`nvtiffDecodeImageEx
  code 4` → silent CPU fallback, 65 p/s). Not viable for this format.

**Net: it's a storage ↔ throughput tradeoff.** Uncompressed buys compute-bound
(1,203 p/s) at large storage; deflate stays CPU-class (~489) unless someone
writes the fused decode kernel.

### GPU-decode alternatives evaluated (so colleagues don't re-tread)

| Approach | Handles 8-band uint16? | Ratio | GPU decode p/s | Verdict |
|---|---|---|---|---|
| torchvision.io (`decode_image`/nvJPEG) | ✗ — JPEG/PNG/WEBP, ≤4-ch uint8 | — | — | can't open our data |
| NVIDIA DALI GPU decoders | routes to nvImageCodec/nvTIFF; TIFF→CPU | — | — | no unlock for TIFF; orchestration only |
| nvTIFF (DEFLATE+predictor) | reads format, **GPU decode fails (code 4)** | 2.37× | 65 (CPU fallback) | unsupported codec combo |
| nvCOMP Deflate (hand-rolled) | ✓ | 2.37× | 489 | gather/predictor-bound ≈ CPU |
| nvJPEG2000 **lossless** | 4-band only (8-band → 2 JP2/patch) | **3.60×** | 136 | best ratio, *slowest* decode |
| **uncompressed** | ✓ | 1.00× | **1,203** | only off-the-shelf compute-bound path |

**The structural finding:** for 8-band uint16, *compression ratio and GPU-decode
speed are inversely related*. Well-compressing codecs (ZSTD, JP2-lossless) decode
slowly or aren't GPU-supported; the GPU-friendly one (Deflate/nvCOMP) is
gather-bound. DALI/torchvision are built for ≤4-channel uint8 JPEG/PNG and don't
apply. So the compact-and-fast quadrant is empty *off-the-shelf* — it requires
the §6a fused custom kernel.

## 7. The decision table — storage × runtime × cost

Full global pass, 98 M patches (20% overlap):

| Codec | Storage | S3 $/mo | Decode | patch/s | 1×H100 | 8-GPU node |
|---|---|---|---|---|---|---|
| Uncompressed (rasterio/CPU) | 265 TB | $6,095 | CPU | 517 | 53 h | 6.6 h |
| Uncompressed (GPUDirect, ideal) | 265 TB | $6,095 | — | 1,426 | 19 h | 2.4 h |
| ZSTD-9 (current) | 135 TB | $3,094 | CPU | 529 | 51 h | 6.4 h |
| DEFLATE+pred (CPU) | 112 TB | $2,572 | CPU | 516 | 53 h | 6.6 h |
| DEFLATE+pred pre-tiled (GPU/nvCOMP, compiled) | 175 TB | $4,020 | GPU | 489 | 56 h | 7.0 h |
| **★ Uncompressed pre-tiled (GPU, compiled)** | **413 TB** | **$9,500** | — | **1,203** | **23 h** | **2.9 h** |
| DEFLATE pre-tiled + fused kernel (TARGET) | 175 TB | $4,020 | GPU | ~1,350 | ~20 h | ~2.5 h |

*(All GPU rows are **measured** except the last (target). Pre-tiled storage
carries ~1.56× overlap redundancy (hence 175/413 TB). 8-GPU CPU rows assume 48
cores/GPU — optimistic; real nodes ~24, so ~2× slower. The DEFLATE-GPU path is
gather/predictor-bound at ~CPU level; only uncompressed reaches compute-bound
off-the-shelf — at a steep storage cost. The compact+fast cell (175 TB,
~1,350 p/s) needs the §6a fused-kernel work.)*

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

## 10. Recommendation (measured)

The model+inference side is solved (`torch.compile` → 95% of ceiling). The whole
problem is getting pixels onto the GPU without a decode tax. Choose by priority:

**A. Max throughput, off-the-shelf today — uncompressed pre-tiled.**
Store overlapping 512² patches uncompressed (NHWC, GSD baked in); read straight
to GPU, `torch.compile` inference. **Measured 1,203 p/s (84% ceiling, 2.6× CPU),
~2.9 h/global-pass on one 8-GPU node.** Cost: ~413 TB storage (~$9.5k/mo S3).
*(Storage-saving variant: uncompressed non-overlap scenes (265 TB) + GPU
`unfold` — untested, load is still free but adds the unfold copy.)*

**B. Compact storage, accept CPU-class speed — DEFLATE+predictor.**
112 TB (~$2.6k/mo), but throughput ~460–530 p/s whether decoded on CPU or GPU
(the GPU gather/predictor glue is the wall). ~6.5 h/8-GPU node.

**C. Compact AND fast — needs engineering.** DEFLATE pre-tiled + a **fused CUDA
decode kernel** (nvCOMP decode → contiguous → predictor in one pass) targets
~1,350 p/s at ~175 TB. This is the ideal, but it's a real project (the off-the-
shelf nvCOMP Python path and nvTIFF do **not** get there — see §6a).

Pipeline regardless of A/B/C: read patch/scene → GPU → (decode if compressed) →
channels_last → bf16 `torch.compile` inference → argmax/polygonize. Feed via
**FSx for Lustre** (the dataset won't fit on local NVMe); keep bucket + instance
**same region** ($0 egress).

**Open follow-ups:** (1) write the §6a fused decode+predictor kernel to make the
compact path compute-bound; (2) measure uncompressed non-overlap scenes + GPU
unfold (variant A at 265 TB); (3) TensorRT/FP8 to lift the compute ceiling above
1,426; (4) re-test nvTIFF on a future release / on Blackwell's HW decompression
engine.

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
