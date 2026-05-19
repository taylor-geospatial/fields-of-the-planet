# ftw-planet — Claude instructions

PlanetScope companion dataset for Fields of the World (FTW) v2, plus field-boundary segmentation training/eval. Higher-res (3 m) version of FTW S2 patches.

## Stack

- Python 3.13 (project pins `>=3.13`).
- Package manager: `uv`. Lint: `ruff`. Type-check: `ty`. Tests: `pytest` (+`pytest-cov`).
- Pre-commit configured.

Make targets:

```
make install   # uv sync --all-extras
make check     # uv run pre-commit run --all-files
make test      # uv run pytest --cov=src tests/
make clean
```

## Layout

- `src/ftw_planet/` — package. Modules: `datasets.py`, `datamodules.py`, `trainers.py`, `losses.py`, `pipeline.py`, `planet.py`, `ftw.py`.
- `scripts/` — pipeline (`search_shard.py`, `activate_global.py`, `extract_shard.py`, `resample.py`, `udm2_fill.py`, `rasterize_labels.py`, ...) + `train.py` / `eval_planet.py` / `polygon_metrics_eval.py` / `postprocess_eval.py` / `viz_predictions.py`.
- `scripts/slurm/*.sbatch` — SLURM wrappers, one per phase.
- `configs/prue/*.yaml` — Hydra/Lightning configs for training (PRUE = Planet R UNet Experiments).
- `data/` — gitignored. Planet artifacts under `data/planet/<country>/`, `_global/` for manifests + caches.
- `logs/` — gitignored. W&B runs + checkpoints under `logs/prue/<run_name>/`.
- `paper/` — LaTeX source. Owned by other agents — do not touch unless asked.
- `tests/` — pytest suite.

## Conventions

- **NEVER** `from __future__ import annotations`. Forbidden.
- Keep files <~500 LOC. Split/refactor when growing.
- Conventional Commits: `feat|fix|refactor|build|ci|chore|docs|style|perf|test`.
- **No silent failures.** Don't wrap suspicious code in `try/except → warn → return None/NaN/[]`. Use narrowest exception type; if you don't know it, run and find out. No `# noqa` / `# type: ignore` without a one-line justification. (See user global CLAUDE.md for full rationale.)
- Field-boundary metrics: lead with **object F1 / polygon quality (PQ/SQ/RQ, AP)**. Pixel IoU is secondary.
- Ruff config in `pyproject.toml` is strict (ANN, B, TRY, UP, ...). Respect per-file ignores already there for `scripts/`, `paper/`, `tests/`.

## HPC (RAILS)

- SLURM account: `--account=bgtj-tgirails` (already wired into sbatch templates).
- CPU/IO work: submit to `cpu` or `cpu_amd` partition. `cpu_amd` is cheapest. Do **not** burn GPU nodes on pipeline phases.
- Training/eval: `gpu` partition. Template: `scripts/slurm/train_prue.sbatch`.
- All pipeline phases have sbatch wrappers in `scripts/slurm/`. Submit with `--dependency=afterok:$PREV` to chain.

## Pointers

- `AGENTS.md` — handoff overview for other agents.
- `scripts/README.md` — full 11-phase pipeline doc, dependency chain, knobs.
- `docs/profiling.md` — per-phase wall-clock + throughput from the 140k-patch run.
- `docs/planet-api-issues.md` — Planet-side reliability gotchas (broken activation URLs, cold-storage thaw tails, regional dead assets, 429 patterns).
- `README_S3.md` — S3 upload/release notes.
- Skills: `.claude/skills/ftw-planet-pipeline/` (pipeline ops), `.claude/skills/ftw-planet-training/` (training/eval). User-level `planet-bulk-download` skill has the distilled Planet API playbook.
