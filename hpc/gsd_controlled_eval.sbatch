#!/bin/bash
#SBATCH --job-name=ftw-gsd
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gsd_%j.out
#SBATCH --error=logs/gsd_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PLANET_CKPT="${PLANET_CKPT:-logs/prue/ftw_planet-unet-efnet3-crop512-v3-augmax-full/ftw-planet/mt6mdnl7/checkpoints/last.ckpt}"
S2_CKPT="${S2_CKPT:-logs/prue/ftw_s2-unet-efnet7-crop256-s2-v3-augmax-b7-full/ftw-s2/2x26jpwu/checkpoints/last.ckpt}"
OUT="${OUT:-logs/gsd_controlled/heldout11.csv}"
SPLIT="${SPLIT:-test}"

mkdir -p "$(dirname "$OUT")"

uv run --no-sync python scripts/eval/gsd_controlled_eval.py \
    --planet-ckpt "$PLANET_CKPT" \
    --s2-ckpt "$S2_CKPT" \
    --out "$OUT" \
    --split "$SPLIT" \
    --gpu 0
