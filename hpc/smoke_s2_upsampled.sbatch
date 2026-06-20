#!/bin/bash
#SBATCH --job-name=ftw-s2-up-smoke
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_s2_up_%j.out
#SBATCH --error=logs/smoke_s2_up_%j.err

# Smoke test for the resized-S2 control: tiny run to confirm 512x512 shapes,
# the upsample transform, and that training steps run on the A100.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONFIG="configs/prue/ftw_s2_efnet3_crop512_upsampled_augmax_full.yaml"

nvidia-smi || true

# `--` separates click CLI_ARGS from ftw's LightningCLI args.
uv run --no-sync ftw model fit -c "$CONFIG" -- \
    --trainer.limit_train_batches=2 \
    --trainer.limit_val_batches=2 \
    --trainer.max_epochs=1 \
    --trainer.logger=false \
    --trainer.default_root_dir=logs/prue/_smoke_s2_upsampled
