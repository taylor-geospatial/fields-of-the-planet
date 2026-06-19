#!/bin/bash
#SBATCH --job-name=ftw-s2-up
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Resized-S2 control (reviewer request): EfficientNet-B3 UNet on FTW Sentinel-2
# with inputs bilinear-upsampled 256->512 (mask nearest-upsampled), same recipe
# as the native S2 B3 augmax-full run. Tests whether the Planet>S2 gap is real
# 3m detail vs. just bigger crop / more pixels.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONFIG="${CONFIG:-configs/prue/ftw_s2_efnet3_crop512_upsampled_augmax_full.yaml}"
CKPT_PATH="${CKPT_PATH:-}"

nvidia-smi || true

if [ -n "$CKPT_PATH" ]; then
    echo "Resuming from checkpoint: $CKPT_PATH"
    uv run --no-sync ftw model fit -c "$CONFIG" --ckpt_path "$CKPT_PATH"
else
    uv run --no-sync ftw model fit -c "$CONFIG"
fi
