#!/bin/bash
#SBATCH --job-name=ftw-prue
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# PRUE EfficientNet-B3 UNet on FTW-Planet (PlanetScope SR + FTW v2 labels).
# Single H100, bf16-mixed, batch 64, random 256x256 crops.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONFIG="${CONFIG:-configs/prue/ftw_planet_efnet3.yaml}"

nvidia-smi || true

uv run --no-sync ftw model fit -c "$CONFIG"
