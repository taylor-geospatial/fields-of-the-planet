#!/bin/bash
#SBATCH --job-name=ftw-s2up-b7
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/s2up_b7_%j.out
#SBATCH --error=logs/s2up_b7_%j.err

# Re-score the EXISTING released PRUE B7 S2 checkpoint with --upsample-to 512
# (bilinear 256->512, mask nearest), matching ftw_tools' resize_factor=2 — the
# resolution at which PRUE's released numbers were produced. This is the
# canonical fix for the 512-zero-pad bug, NOT native 256. Mirrors the B3
# upsampled control (logs/polygon_metrics/s2_upsampled_b3_augmax_full.csv).
#
# Only --upsample-to differs from the shipped runs; ckpt, TTA, watershed, and
# --s2-data-scale 3000 are held fixed.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/per_patch logs/polygon_metrics

S2_CKPT="${S2_CKPT:-logs/best_checkpoints/s2_efnet7_best.ckpt}"

# Per-patch: 10 dense-label held-out countries (feeds the improvement figure).
DENSE10="belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden"
# Headline polygon metrics: canonical FTW 22-country test split.
FTW22="austria belgium brazil cambodia corsica croatia denmark estonia finland france germany latvia lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam"

PERPATCH_OUT="logs/per_patch/s2_b7_upsampled512.csv"
HEADLINE_OUT="logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv"
rm -f "$PERPATCH_OUT" "$HEADLINE_OUT"

# shellcheck disable=SC2206
DENSE10_ARGS=( $DENSE10 )
# shellcheck disable=SC2206
FTW22_ARGS=( $FTW22 )

echo "=== S2 B7 per-patch @ upsample 512 (10 dense held-out) ==="
uv run --no-sync python scripts/eval/per_patch_metrics.py \
    --ckpt "$S2_CKPT" \
    --out "$PERPATCH_OUT" \
    --root data/ftw \
    --dataset-backend s2 --upsample-to 512 \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 512 \
    --countries "${DENSE10_ARGS[@]}"

echo "=== S2 B7 headline polygon metrics @ upsample 512 (FTW22) ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt "$S2_CKPT" \
    --out "$HEADLINE_OUT" \
    --root data/ftw \
    --dataset-backend s2 --upsample-to 512 \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 512 \
    --countries "${FTW22_ARGS[@]}"

echo "done. wrote:"
echo "  $PERPATCH_OUT"
echo "  $HEADLINE_OUT"
