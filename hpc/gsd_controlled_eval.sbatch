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

PLANET_CKPT="${PLANET_CKPT:-logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt}"
S2_CKPT="${S2_CKPT:-logs/best_checkpoints/s2_efnet7_best.ckpt}"
# resize_factor=2 upsamples the S2 input 256->512 before inference, matching the
# headline upsample-512 S2 eval. Output is tagged so it sits beside any native run.
S2_RESIZE_FACTOR="${S2_RESIZE_FACTOR:-2}"
OUT="${OUT:-logs/gsd_controlled/heldout11_rf${S2_RESIZE_FACTOR}.csv}"
SPLIT="${SPLIT:-test}"

mkdir -p "$(dirname "$OUT")"

uv run --no-sync python scripts/eval/gsd_controlled_eval.py \
    --planet-ckpt "$PLANET_CKPT" \
    --s2-ckpt "$S2_CKPT" \
    --out "$OUT" \
    --split "$SPLIT" \
    --s2-resize-factor "$S2_RESIZE_FACTOR" \
    --gpu 0
