#!/bin/bash
#SBATCH --job-name=ftw-perpatch
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/perpatch_%j.out
#SBATCH --error=logs/perpatch_%j.err

# Per-patch polygon metrics for BOTH models over the 10 dense held-out
# countries. Writes logs/per_patch/planet_b3.csv and logs/per_patch/s2_b7.csv.
#
# Smoke-test one country first by setting COUNTRIES=croatia before the full run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/per_patch

# 10 dense-label held-out countries (HELDOUT_10_DENSE; excludes presence-only kenya).
COUNTRIES="${COUNTRIES:-belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden}"
# shellcheck disable=SC2206  # intentional word-splitting
COUNTRY_ARGS=( $COUNTRIES )

PLANET_CKPT="${PLANET_CKPT:-logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt}"
S2_CKPT="${S2_CKPT:-logs/best_checkpoints/s2_efnet7_best.ckpt}"
PLANET_OUT="${PLANET_OUT:-logs/per_patch/planet_b3.csv}"
S2_OUT="${S2_OUT:-logs/per_patch/s2_b7.csv}"

# Fresh CSVs each run (the script appends).
rm -f "$PLANET_OUT" "$S2_OUT"

echo "=== Planet B3 (planet backend) ==="
uv run --no-sync python scripts/eval/per_patch_metrics.py \
    --ckpt "$PLANET_CKPT" \
    --out "$PLANET_OUT" \
    --root data \
    --dataset-backend planet \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 512 \
    --countries "${COUNTRY_ARGS[@]}"

echo "=== S2 B7 (s2 backend) ==="
uv run --no-sync python scripts/eval/per_patch_metrics.py \
    --ckpt "$S2_CKPT" \
    --out "$S2_OUT" \
    --root data/ftw \
    --dataset-backend s2 \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 512 \
    --countries "${COUNTRY_ARGS[@]}"

echo "done."
