---
name: ftw-planet-pipeline
description: Operate the FTW v2 → PlanetScope dataset pipeline on SLURM. Covers the 11-phase search/activate/extract/rasterize chain, SLURM submission with dependencies, resample/UDM2-fill recovery loops, and Planet API failure modes (cold-storage thaw, broken activation URLs, 429s). Use when the user asks to run/rerun/extend the pipeline, recover failed patches, profile a phase, or diagnose Planet-side errors.
---

# FTW-Planet Pipeline

Pairs FTW v2 S2 patches with PlanetScope SR + UDM2, rasterizes per-country polygons onto each patch's native UTM grid. Output: `data/planet/<country>/<patch_id>_<window>{.tif,_udm2.tif,_label.tif}`.

All phases have a Python script in `scripts/` and a SLURM wrapper in `scripts/slurm/`. Submit from `scripts/`.

## Phases

| #   | Script                       | sbatch                          | Purpose                                                                                       |
| --- | ---------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | `download_ftw.py`            | `download_ftw.sbatch`           | Pull FTW v2 country tars + write `data/ftw/<country>/index.jsonl`.                            |
| 2   | `download_polygons.py`       | — (local)                       | Per-country GeoParquet polygons from Source Cooperative → `data/ftw_polygons/<country>.parquet`. |
| 3   | `build_manifest.py`          | `manifest.sbatch`               | Merge all `index.jsonl` → `data/planet/_global/manifest.jsonl` (one row per country/patch/window). |
| 4   | `search_shard.py`            | `search.sbatch` (array)         | Sharded Planet Data API search. Writes `_global/search/shard_NNN.jsonl`.                      |
| 5   | `activate_global.py`         | `activate.sbatch`               | Dedup item_ids, activate SR + UDM2 in parallel. Writes `_global/activations.jsonl`.           |
| 6   | `extract_shard.py`           | `extract.sbatch` (array)        | Scene-grouped extract. One COG opened once via `/vsicurl/`, range-reads each window's SR+UDM2. |
| 7   | `udm2_quality.py`            | `udm2_quality.sbatch`           | Per-patch UDM2 band stats. **OVERWRITES** `_global/udm2_quality.jsonl` each run.              |
| 8   | `resample.py`                | `resample.sbatch`               | 3-phase recovery (search-alts → UDM2 probe → SR-only extract) for failed/broken patches.      |
| 9   | `udm2_fill.py`               | `udm2_fill.sbatch`              | Fill SR patches missing UDM2 companion (plan → activate → extract).                           |
| 10  | `rasterize_labels.py`        | `rasterize_labels.sbatch` (array) | Rasterize polygons → 3-class label tif (0=bg, 1=field, 2=boundary) on each SR patch's grid.   |
| 11  | `prune_singleton_patches.py` | —                               | Remove patches missing one of the two seasonal windows. Dry-run by default; `--apply` to act. |

Re-run `udm2_quality.sbatch` after resample+fill to re-score swapped-in patches.

## Full dependency chain

```bash
cd scripts
DL=$(sbatch --parsable slurm/download_ftw.sbatch)
MAN=$(sbatch --parsable --dependency=afterok:$DL    slurm/manifest.sbatch)
SEARCH=$(sbatch --parsable --dependency=afterok:$MAN  slurm/search.sbatch)
ACT=$(sbatch --parsable --dependency=afterok:$SEARCH slurm/activate.sbatch)
EXTRACT=$(sbatch --parsable --dependency=afterok:$ACT slurm/extract.sbatch)
QUAL=$(sbatch --parsable --dependency=afterok:$EXTRACT slurm/udm2_quality.sbatch)
RESAMPLE=$(sbatch --parsable --dependency=afterok:$QUAL slurm/resample.sbatch)
FILL=$(sbatch --parsable --dependency=afterok:$RESAMPLE slurm/udm2_fill.sbatch)
LABELS=$(sbatch --parsable --dependency=afterok:$FILL slurm/rasterize_labels.sbatch)
QUAL2=$(sbatch --parsable --dependency=afterok:$LABELS slurm/udm2_quality.sbatch)
```

`download_polygons.py` runs locally before phase 10 — not in chain. `prune_singleton_patches.py` is manual cleanup after labels.

## Knobs (env vars at submit)

- `MAX_CC` — cloud-cover ceiling for search (default 0.1).
- `SC` / `AC` / `EC` — search / activate / extract concurrency.
- `MAX_CANDS` — resample candidates per patch.
- Country slug — passed to `download_ftw.py` and `rasterize_labels.py`.

```bash
MAX_CC=0.05 SC=64 sbatch slurm/search.sbatch
MAX_CANDS=10  sbatch slurm/resample.sbatch
```

## Failure modes & recovery

- **Cold-storage activations** dominate phase 5 wall time. Mean ~46 s, **p100 ~244 s** observed on Rwanda; p99 globally >2700 s. Do not kill long activate jobs — they are thawing scenes from Planet cold storage. Warmed scenes re-serve in ~1.6 s.
- **Broken activation URLs (~23% rate).** Planet sometimes returns 200 OK with a download URL that 404/410s on read. `resample.py` (phase 8) is the recovery path: searches alternative scenes covering the same window, re-activates, re-extracts SR-only.
- **SR without matching UDM2.** Common when only one asset type activates cleanly. `udm2_fill.py` (phase 9) plans+activates+extracts the companion UDM2 for any SR patch missing one.
- **Singleton windows.** FTW training requires both seasonal windows. `prune_singleton_patches.py --apply` removes patches with only one. Run after labels.
- **429s.** Planet rate limits. Drop `SC`/`AC`/`EC`. See `docs/planet-api-issues.md`.
- **Phase-7 file overwrite.** `udm2_quality.py` overwrites — not append. Always re-run after recovery phases.

## Profiling

`docs/profiling.md` has per-phase wall-clock + throughput numbers from the 140k-patch run. Activation is the bottleneck (~7–12 hr at full scale, 16-way). Search ~40 min. Extract ~55 min.

## Planet-side gotchas

`docs/planet-api-issues.md` — broken activation URLs, cold-storage thaw tails, region-specific dead assets, 429 patterns. User-level skill `planet-bulk-download` has the distilled architecture (search/activate/extract, COG `/vsicurl/` range-reads vs Orders API, GDAL config, retry budgets).

## Before running — checklist

- [ ] `.env` at repo root with `PL_API_KEY=...` (loaded via `python-dotenv` in scripts).
- [ ] SLURM account: sbatch templates already set `--account=bgtj-tgirails`. Confirm.
- [ ] Output dirs writable: `data/planet/`, `data/ftw/`, `data/ftw_polygons/`, `logs/`.
- [ ] Polygons downloaded locally (`download_polygons.py`) before phase 10.
- [ ] Submit pipeline phases to `cpu` or `cpu_amd` partition — never GPU. Sbatch defaults should already reflect this; double-check `--partition=` line.
- [ ] If extending to a new country: phase-1 country slug must match FTW v2 naming. Check `data/ftw/<slug>/index.jsonl` exists before manifest.

## Artifact layout

```
data/planet/
├── _global/
│   ├── manifest.jsonl
│   ├── search/shard_NNN.jsonl
│   ├── activations.jsonl
│   ├── extract/shard_NNN.jsonl
│   ├── udm2_quality.jsonl
│   ├── resample/{search,probes,picks,sr_activations}.jsonl
│   ├── udm2_fill/{plan,activations,extracts}.jsonl
│   ├── rasterize_summary.jsonl
│   └── resample_log.jsonl
└── <country>/
    ├── <patch_id>_a.tif         # 4-band SR uint16, UTM, ~3 m
    ├── <patch_id>_a_udm2.tif    # 8-band UDM2 uint8
    ├── <patch_id>_a_label.tif   # 3-class label uint8
    └── <patch_id>_b.tif, ...
```
