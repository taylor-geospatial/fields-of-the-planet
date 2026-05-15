#!/bin/bash
#SBATCH --job-name=ftw-eval
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CKPT="${CKPT:-logs/prue/ftw_planet-unet-efnet3-bf16/ftw-planet/1io8addz/checkpoints/last.ckpt}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-logs/eval_planet_${SPLIT}.csv}"

uv run --no-sync python scripts/eval_planet.py \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --split "$SPLIT" \
    --model-predicts-3-classes \
    --gpu 0 \
    --num-workers 8
