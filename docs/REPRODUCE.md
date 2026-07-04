# Reproducing FTP training and evaluation

This guide reruns the Fields of the Planet (FTP) training and evaluation
behind a single command line. It assumes a machine with one CUDA GPU and the
dataset already present under `data/` (see the Dataset section in the
[README](../README.md#dataset)).

## Install

```bash
make install        # uv sync --all-extras
```

This installs the `ftw-planet` command. Run all commands from the repository
root.

## 1. Check the data

```bash
ftw-planet check-data --split dense10
```

Confirms `data/planet/<country>/` exists for the evaluation countries and
prints the expected layout if anything is missing.

## 2. Evaluate the released checkpoint

```bash
ftw-planet reproduce --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt
```

Runs watershed + D4 TTA inference over the 10 dense-label held-out countries,
writes per-country CSVs to `logs/eval/<checkpoint>/`, and prints the macro
table next to the published paper numbers:

| Metric          | Expected |
| --------------- | -------- |
| PQ              | 0.355    |
| Obj F1 (WS+TTA) | 0.452    |
| Pixel IoU       | 0.688    |

Presence-only Kenya is excluded from these aggregates; pass `--split full23`
for the 23-region full-data protocol.

## 3. Train from scratch

```bash
ftw-planet train configs/prue/ftw_planet_efnet3_crop512_v3_augmax_full.yaml
```

Trains the EfficientNet-B3 U-Net with the augmax recipe (seed 7). Checkpoints
land under `logs/prue/<run>/`. Resume with `--resume <last.ckpt>`. To train
and immediately evaluate:

```bash
ftw-planet reproduce \
  --train configs/prue/ftw_planet_efnet3_crop512_v3_augmax_full.yaml \
  --ckpt logs/prue/<run>/checkpoints/last.ckpt
```
