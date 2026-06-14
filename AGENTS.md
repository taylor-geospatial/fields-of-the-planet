# AGENTS.md

Notes for Claude Code / coding agents working in this repo.

## What this is

PlanetScope (~3 m) companion to **Fields of the World v2**. Build paired SR+UDM2 imagery for FTW patches across 25 countries; train segmentation models for ~3 m field boundaries. Pipeline + package + paper in one repo.

## Stack

- Python 3.13, [uv](https://docs.astral.sh/uv/) for deps
- ruff (format + lint), ty (types), pytest + pytest-cov, pre-commit
- lightning + torchgeo-style trainers; LightningCLI YAML configs; training via `ftw model fit -c <config>` (`ftw-tools`)
- Planet Python SDK v2, rasterio, geopandas, pyarrow

## Commands

```bash
make install   # uv sync --all-extras
make check     # pre-commit run --all-files (ruff, ty, ...)
make test      # pytest --cov=src tests/
make clean
```

Run anything Python via `uv run ...` — never plain `python`.

## Layout

```
src/ftw_planet/   datamodules, datasets (FTWPlanet), trainers, losses, planet/pipeline helpers
scripts/eval/     evaluation entrypoints
scripts/pipeline/ dataset-building pipeline
hpc/              optional SLURM wrappers, one per pipeline phase
configs/prue/     LightningCLI-style YAML configs (efnet3/5/7, crop sizes, loss variants)
data/             FTW patches + Planet outputs (gitignored)
logs/             checkpoints + W&B (gitignored)
paper/            LaTeX source
docs/             profiling.md, planet-api-issues.md
```

## Pipeline

End-to-end search → activate → extract → rasterize → prune. Full phase table, artifact layout, and SLURM invocations in [`scripts/README.md`](scripts/README.md). Don't duplicate it.

## Conventions

- **Never** `from __future__ import annotations`.
- Keep files \<500 LOC. Split / refactor when they grow.
- Conventional Commits (`feat|fix|refactor|build|ci|chore|docs|style|perf|test`).
- Telegraph style in docs. No filler.
- No silent `try/except` to swallow failures — narrow exception types, fix root cause. Same for `# type: ignore` / `# noqa`.
- Run `make check` before handoff. CI red → `gh run list/view`, fix, repeat til green.

## Metrics

Field-boundary work: **lead with object F1 + polygon metrics**. Pixel IoU is secondary. See `scripts/eval/polygon_metrics_eval.py`, `scripts/eval/eval_planet.py`.

## HPC notes (TGI RAILS cluster)

- SLURM account: `--account=bgtj-tgirails` (required on all sbatch/srun).
- Heavy I/O / pipeline jobs (search, activate, extract, rasterize): prefer `--partition=cpu` or `--partition=cpu_amd`. Keep them off GPU partitions.
- Training: GPU partition. Configs default to `bf16-mixed`.
- Planet downloads: watch for 429s and broken-URL activations — see `docs/planet-api-issues.md`.

## Pointers

- Pipeline: [`scripts/README.md`](scripts/README.md)
- Planet API gotchas: [`docs/planet-api-issues.md`](docs/planet-api-issues.md)
- Profiling notes: [`docs/profiling.md`](docs/profiling.md)
- Dataset card: see "Dataset" section in root `README.md`
