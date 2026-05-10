# FTW → PlanetScope match — pipeline profiling

Wall-clock numbers for the three-phase pipeline against **Fields of the World v2 / Rwanda** (70 patches × 2 windows = 140 extracts), running on a single `cpu_amd` SLURM node (16 CPUs, 32 GB RAM). Data: PSScene `ortho_analytic_4b_sr` (4-band SR, ~3 m, COG) + `ortho_udm2`. Cloud cover ≤ 0.1, full-coverage scenes only. Date 2026-05-09.

## Per-stage profile

| Stage                                       | Concurrency |              Wall | Per-call median | Per-call max | Notes                                            |
| ------------------------------------------- | ----------: | ----------------: | --------------: | -----------: | ------------------------------------------------ |
| **Search** (Data API)                       |          32 |            14.8 s |          0.40 s |       13.3 s | 140 metadata calls                               |
| **Activate** (cold-storage thaw + sign URL) |          16 |    414 s (~7 min) |           1.6 s |    **244 s** | 16 unique scenes; mean 46 s                      |
| **Extract — network read**                  |          32 |              11 s |          0.51 s |       0.91 s | HTTP range read SR (~1 MB transferred per patch) |
| **Extract — local disk write**              |          32 | (subset of above) |          0.08 s |       0.33 s | ZSTD-compressed GeoTIFF                          |
| **UDM2 read**                               |           — |                 — |          0.18 s |       0.73 s | smaller asset (~30 KB out)                       |
| **UDM2 write**                              |           — |                 — |          0.03 s |       0.16 s | —                                                |

Total wall for full Rwanda (cold cache): **7 min 22 s**, dominated entirely by Phase 2.

## Key findings

1. **Activation is the bottleneck.** Median 1.6 s for warm scenes vs **244 s p100** for cold-storage thaw. Mean 46 s. Once a scene is active, the URL is good for ~24 h and re-activation is sub-second.
1. **Writes are ~6× faster than reads** (0.08 s vs 0.51 s for SR). Splitting read/write into separate worker pools would not help — network is the only bottleneck. Just bump extract concurrency.
1. **Scene reuse: 8.8 patches per scene in Rwanda.** 16 unique PSScene COGs covered all 140 windows. The activation cache amortizes cleanly over many extracts.
1. **Range reads are cheap.** ~1 MB transferred per SR patch out of a ~500 MB scene strip → COG range reads are working as intended. No reason to involve the Orders API.

## Extrapolation to full FTW (~75k patches × 2 windows = 150k extracts)

Assumptions:

- Rwanda's 8.8 patches/scene is optimistic for global; bracket as **8.5k–15k unique scenes**.
- Cold-storage scenes dominate first hit; warmed scenes serve from ~1.6 s.

Wall-time at current concurrency (16 / 32):

| Stage                 | Time                    |
| --------------------- | ----------------------- |
| Search (32-way)       | ~40 min                 |
| **Activate (16-way)** | **~7–12 hr ← dominant** |
| Extract (32-way)      | ~55 min                 |
| **Total**             | **~9–14 hr**            |

If activate concurrency goes to 64 (assuming Planet does not throttle): activation drops to **~2–3 hr**, total **~4–5 hr**.

## Levers for the supervisor / Planet conversation

The numbers worth quoting:

- We will need ~10–15k unique PSScene activations for the FTW global v2 corpus.
- Median activation latency 1.6 s but **p99 ≥ 4 minutes** — cold-storage thaw is the wall.
- The download itself is a non-issue: ~1 MB per patch × 150k = ~150 GB total, served as parallel HTTP range reads from a CDN.
- What would actually help (in priority order):
    1. **Bulk pre-warm a list of scene IDs** ahead of the run so we skip cold-storage thaw entirely.
    1. **Higher per-account activation rate limit** (we are pacing at 16-way; want 64+).
    1. **Direct GCS bucket access** for the COGs, bypassing `:activate` entirely — turns the whole pipeline into pure GDAL `/vsicurl/` reads.

## What we are NOT bound on

- CPU: 4 of 16 cores busy peak; not a CPU job.
- Disk: writes are 6× faster than network reads; not a disk job.
- Egress bandwidth: range reads pull a few MB per patch; cluster has plenty.
- Local activation polling: the planet SDK's `wait_asset` polls every 5 s; could shave a bit by polling faster, but it is irrelevant next to the cold-thaw wall.

## Cache files (resumable)

Each phase writes JSONL incrementally; re-running skips already-done work:

- `data/planet/<country>/search.jsonl` — per (patch, window) chosen item_id
- `data/planet/<country>/activations.jsonl` — per item_id signed SR + UDM2 URLs
- `data/planet/<country>/extracts.jsonl` — per (patch, window) timing + status

So if a SLURM job times out mid-run, just resubmit. If signed URLs expire (>24 h), delete `activations.jsonl` and re-run only Phase 2 with `--phase activate`.
