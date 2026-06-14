# scripts/

Production pipeline for building a PlanetScope companion dataset to FTW v2.

Layout:

- `scripts/*.py` — production pipeline scripts (run in the order below).
- `hpc/*.sbatch` — production SLURM wrappers (one per phase).

All artifacts land under `data/`. Planet outputs live in `data/planet/`.

## Pipeline phases

Each phase has a Python script and (where applicable) a SLURM wrapper.

| #   | Script                       | SLURM                                   | Purpose                                                                                                                                                                                                                             |
| --- | ---------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `download_ftw.py`            | `hpc/download_ftw.sbatch`             | Download + index one or all FTW v2 countries into `data/ftw/<country>/index.jsonl`.                                                                                                                                                 |
| 2   | `download_polygons.py`       | — (local)                               | Download per-country GeoParquet polygons from Source Cooperative into `data/ftw_polygons/<country>.parquet`.                                                                                                                        |
| 3   | `build_manifest.py`          | `hpc/manifest.sbatch`                 | Combine every country's `index.jsonl` into `data/planet/_global/manifest.jsonl`, one row per (country, patch, window).                                                                                                              |
| 4   | `search_shard.py`            | `hpc/search.sbatch` (array)           | Sharded Planet Data API search. Each task processes its slice of the manifest, writes `_global/search/shard_NNN.jsonl`.                                                                                                             |
| 5   | `activate_global.py`         | `hpc/activate.sbatch`                 | Dedup item_ids across search shards, call Planet `:activate` for SR + UDM2 in parallel, write `_global/activations.jsonl`.                                                                                                          |
| 6   | `extract_shard.py`           | `hpc/extract.sbatch` (array)          | Scene-grouped extract. Each task owns scenes via `hash(item_id) % num_shards`, opens each COG once via `/vsicurl/`, range-reads every member patch's window. Writes `<country>/<id>_<window>.tif` for SR + UDM2.                    |
| 7   | `udm2_quality.py`            | `hpc/udm2_quality.sbatch`             | Compute per-patch UDM2 band stats (clear / cloud / shadow / haze / snow / unusable). Writes `_global/udm2_quality.jsonl`. **OVERWRITES** the file each run (not append).                                                            |
| 8   | `resample.py`                | `hpc/resample.sbatch`                 | 3-phase resample (search-alts → optional UDM2 probe → SR-only extract) for patches failing UDM2 quality thresholds OR with missing/broken-URL extracts. Caches under `_global/resample/{search,probes,picks,sr_activations}.jsonl`. |
| 9   | `udm2_fill.py`               | `hpc/udm2_fill.sbatch`                | 3-phase UDM2 fill (plan → activate → extract) for SR patches lacking their UDM2 companion. Caches under `_global/udm2_fill/{plan,activations,extracts}.jsonl`.                                                                      |
| 10  | `rasterize_labels.py`        | `hpc/rasterize_labels.sbatch` (array) | For each Planet SR tif, query the country's GeoParquet polygons and rasterize onto the patch's native UTM grid as 3-class (0=bg, 1=field, 2=boundary). Writes `<id>_<window>_label.tif`.                                            |
| 11  | `prune_singleton_patches.py` | —                                       | Remove patches with only one of the two seasonal windows (FTW training requires both). Dry-run by default; `--apply` to actually delete.                                                                                            |

## Artifact layout

```
data/planet/
├── _global/
│   ├── manifest.jsonl               # phase 3: (country, patch, window, geom, dates)
│   ├── search/shard_NNN.jsonl       # phase 4
│   ├── activations.jsonl            # phase 5 (item_id -> SR/UDM2 URLs)
│   ├── extract/shard_NNN.jsonl      # phase 6 (per-patch extract status)
│   ├── udm2_quality.jsonl           # phase 7 (per-patch UDM2 stats)
│   ├── resample/                    # phase 8 caches
│   ├── udm2_fill/                   # phase 9 caches
│   ├── rasterize_summary.jsonl      # phase 10
│   └── resample_log.jsonl           # patch-level resample outcomes
├── <country>/
│   ├── <patch_id>_a.tif             # 4-band SR (uint16, UTM, ~3m)
│   ├── <patch_id>_a_udm2.tif        # 8-band UDM2 (uint8)
│   ├── <patch_id>_a_label.tif       # 3-class label (uint8)
│   └── <patch_id>_b.tif/...
```

## Reproducing the full pipeline (RAILS, end-to-end)

Submit all phases in dependency order from `scripts/`:

```bash
DL=$(sbatch --parsable slurm/download_ftw.sbatch)
MAN=$(sbatch --parsable --dependency=afterok:$DL slurm/manifest.sbatch)
SEARCH=$(sbatch --parsable --dependency=afterok:$MAN slurm/search.sbatch)
ACT=$(sbatch --parsable --dependency=afterok:$SEARCH slurm/activate.sbatch)
EXTRACT=$(sbatch --parsable --dependency=afterok:$ACT slurm/extract.sbatch)
QUAL=$(sbatch --parsable --dependency=afterok:$EXTRACT slurm/udm2_quality.sbatch)
RESAMPLE=$(sbatch --parsable --dependency=afterok:$QUAL slurm/resample.sbatch)
FILL=$(sbatch --parsable --dependency=afterok:$RESAMPLE slurm/udm2_fill.sbatch)
LABELS=$(sbatch --parsable --dependency=afterok:$FILL slurm/rasterize_labels.sbatch)
QUAL2=$(sbatch --parsable --dependency=afterok:$LABELS slurm/udm2_quality.sbatch)
```

`download_polygons.py` is run locally before phase 10; it is not in the SLURM chain.
`prune_singleton_patches.py` is a manual cleanup step after labels.
The second `udm2_quality` (`QUAL2`) re-scores patches that were swapped in by resample / udm2_fill.

## Knobs

Env vars typically tuned at submit time (see each `.sbatch` for defaults):

- `MAX_CC` — search cloud-cover ceiling (default 0.1).
- `SC` / `AC` / `EC` — search / activate / extract concurrency.
- `MAX_CANDS` — resample candidates per patch.
- Country slug — passed to `download_ftw.py` (phase 1) and `rasterize_labels.py` (phase 10).

Example:

```bash
MAX_CC=0.05 SC=64 sbatch slurm/search.sbatch
MAX_CANDS=10 sbatch slurm/resample.sbatch
```

## Training

`hpc/train_prue.sbatch` wraps `ftw model fit -c <config>`. Override the config
with `CONFIG=...`, resume with `CKPT_PATH=...`, and run a multi-seed sweep with
`SEED=<int>` (overrides `seed_everything` in the YAML and tags the W&B run
name plus `default_root_dir` with `_seed<N>` to keep checkpoints/logs apart):

```bash
CONFIG=configs/prue/ftw_planet_efnet3_crop512_v3_augmax.yaml
for S in 7 13 42; do
    SEED=$S CONFIG=$CONFIG sbatch hpc/train_prue.sbatch
done
```

Mechanism: extra args pass through click's `CLI_ARGS` to LightningCLI after a
`--` separator, so any nested key works (e.g. `--trainer.max_epochs=50`).

## Lessons learned / references

- `docs/profiling.md` — per-phase wall-clock + throughput numbers from the 140k-patch run.
- `docs/planet-api-issues.md` — Planet-side reliability gotchas (broken activation URLs, cold-storage thaw tails, regional dead assets, 429 patterns).
- `/u/isaaccorley/.claude/skills/planet-bulk-download/SKILL.md` — distilled reference for future Planet bulk runs (search/activate/extract architecture, COG range-reads vs Orders API, GDAL config, retry logic, concurrency budgets).
