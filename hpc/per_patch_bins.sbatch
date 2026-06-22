#!/bin/bash
#SBATCH --job-name=ftw-ppbins
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ppbins_%j.out
#SBATCH --error=logs/ppbins_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DENSE10="belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden"

rm -f logs/per_patch/planet_b3_bins.csv logs/per_patch/s2_b7_bins.csv

# Planet B3 (native 3m grid).
uv run --no-sync python scripts/eval/per_patch_metrics.py \
  --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt \
  --out logs/per_patch/planet_b3_bins.csv \
  --dataset-backend planet --countries $DENSE10 \
  --watershed --tta --min-pad-size 512 \
  --area-bins 0.5,2 --pixel-size-m 3

# S2 B7 (upsample-512 -> 5m grid; headline S2 protocol). Root is data/ftw.
uv run --no-sync python scripts/eval/per_patch_metrics.py \
  --ckpt logs/best_checkpoints/s2_efnet7_best.ckpt \
  --out logs/per_patch/s2_b7_bins.csv \
  --root data/ftw --dataset-backend s2 --countries $DENSE10 \
  --watershed --tta --min-pad-size 512 --upsample-to 512 \
  --area-bins 0.5,2 --pixel-size-m 5

echo "done"
