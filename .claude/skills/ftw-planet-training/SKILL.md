______________________________________________________________________

## name: ftw-planet-training description: Train and evaluate field-boundary segmentation models on the FTW-Planet dataset. Covers the LightningCLI config layout under configs/prue/, training via `ftw model fit`, eval scripts (eval_planet.py, polygon_metrics_eval.py, postprocess_eval.py, viz_predictions.py), and the metrics that actually matter (object F1, polygon quality > pixel IoU). Use when the user asks to train a new model, run evals, sweep configs, or interpret metrics.

# FTW-Planet Training & Eval

PRUE = Planet R UNet Experiments. LightningCLI + timm efficientnet encoders, UNet decoder. Targets: 3-class (bg / field / boundary) + optional SDF head.

## Training launch

Training driver is `ftw model fit -c <config>` from `ftw-tools` — the LightningCLI everything is built against. Configs live under `configs/prue/`.

SLURM (preferred):

```bash
# default config
sbatch scripts/slurm/train_prue.sbatch

# pick a config
CONFIG=configs/prue/ftw_planet_efnet3_crop512_v3_augmax_full.yaml \
  sbatch scripts/slurm/train_prue.sbatch

# resume
CKPT_PATH=logs/prue/<run>/checkpoints/last.ckpt \
  CONFIG=configs/prue/<cfg>.yaml \
  sbatch scripts/slurm/train_prue.sbatch
```

Template: H100, `gpu` partition, `bgtj-tgirails` account, bf16-mixed, 24 h walltime, 24 CPUs, 128 G RAM, signal-based requeue on USR1.

Local smoke:

```bash
uv run ftw model fit -c configs/prue/ftw_planet_efnet3.yaml trainer.max_epochs=1
```

## Config layout (`configs/prue/`)

Naming pattern: `<dataset>_<encoder>_<crop>_<variant>.yaml`.

- `ftw_planet_*` — trains on PlanetScope SR (`/10000` norm).
- `ftw_s2_*` — trains on Sentinel-2 patches (`/3000` norm) for the S2 baseline.
- `efnet3` / `efnet5` / `efnet7` — timm `tf_efficientnetv2_{s,m,l}` encoders (B3/B5/B7 family).
- `crop256` / `crop512` — random crop size during training. PlanetScope patches are larger than S2, so 512 is the default for `ftw_planet_*`.
- Variant suffixes:
    - (none) — baseline 3-class CE.
    - `v2_baseline` / `v3_augmax` — augmentation regimes. `v3_augmax_full` = augmax + full FTW (all countries), `v3_augmax_replicate` = subset to match an external baseline, `v3_augmax_sdf` = augmax + SDF head.
    - `sdf` — adds signed distance function regression head (enables watershed at inference).
    - `cldice` — clDice loss (boundary-aware soft-skeletonization).
    - `cutmix` — CutMix augmentation.
    - `framefield` — frame-field head (DECODE-style boundary geometry).
    - `boundary` — extra boundary-class weighting.
    - `curriculum` — staged curriculum learning.
    - `augplus` / `augmax` — incremental aug sets; `augmax_full` is the strongest.

Train config is the union: each YAML sets `trainer.*`, `model.*`, `data.*`, `optimizer.*`. Output dir: `logs/prue/<run_name>/`. W&B project: `ftw-planet`.

## Eval entrypoints

Pick by what you want to measure:

| Script                    | Reports                                                                   | When                                                                  |
| ------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `eval_planet.py`          | Per-country **pixel IoU/precision/recall + object P/R/F1**. CSV.          | Sanity-check checkpoint; matches `ftw-baselines/run_eval.py` schema.  |
| `postprocess_eval.py`     | Object P/R/F1 **with/without** watershed + TTA, side-by-side.             | Postproc ablation (SDF-driven or boundary-EDT watershed, D4 TTA).     |
| `polygon_metrics_eval.py` | **PQ / SQ / RQ**, **AP@[0.5:0.05:0.95]**, polygon-count delta, chamfer-m. | Polygon-level quality. **This is the headline metric** for the paper. |
| `viz_predictions.py`      | Per-patch PNG renders.                                                    | Qualitative inspection.                                               |

Example:

```bash
uv run scripts/polygon_metrics_eval.py \
    --ckpt logs/prue/<run>/checkpoints/last.ckpt \
    --out logs/polygon_metrics/<run>.csv \
    --dataset-backend planet --min-pad-size 512 \
    --watershed --tta
```

SLURM: `scripts/slurm/eval_prue.sbatch`, `scripts/slurm/postproc_eval.sbatch`.

## Metric priority

For field-boundary work, report in this order:

1. **Polygon quality**: PQ / SQ / RQ, AP@[0.5:0.05:0.95] (`polygon_metrics_eval.py`).
1. **Object F1** at IoU 0.5 (`eval_planet.py`, `postprocess_eval.py`).
1. **Pixel IoU** — secondary; useful for training curves, not for headlines.

Pixel IoU rewards blob predictions that fail at the polygon level. Lead with object/polygon metrics.

## Notes

- PlanetScope norm: `/10000`. S2 norm: `/3000`. Don't mix.
- UNet stride-5 encoder requires HxW divisible by 32 — Planet eval pads to nearest multiple, single-sample batch.
- Watershed needs either an SDF head (preferred, DECODE-style) or falls back to `distance_transform_edt` on the predicted boundary class.
- For new training runs, copy the closest `configs/prue/*.yaml`, change `default_root_dir` + W&B `name`, then tweak.
