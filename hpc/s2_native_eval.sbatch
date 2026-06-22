#!/bin/bash
#SBATCH --job-name=ftw-s2-native
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/s2native_%j.out
#SBATCH --error=logs/s2native_%j.err

# Re-run the S2 B7 evals at NATIVE 256 (no upsample, --min-pad-size 256) so we
# can diff against the existing 512-zero-padded CSVs and quantify the padding
# effect. The B7 S2 model was trained at crop256 (config
# ftw_s2_efnet7_crop256_v3_augmax_full.yaml); the shipped eval zero-padded the
# 256 chip into a 512 frame, a train/eval mismatch.
#
# Only --min-pad-size changes vs the original runs (512 -> 256); everything
# else (ckpt, TTA, watershed, --s2-data-scale 3000) is held fixed so padding
# is the only variable.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/per_patch logs/polygon_metrics

S2_CKPT="${S2_CKPT:-logs/best_checkpoints/s2_efnet7_best.ckpt}"

# Per-patch: 10 dense-label held-out countries (feeds the improvement figure).
DENSE10="belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden"
# Headline polygon metrics: canonical FTW 22-country test split.
FTW22="austria belgium brazil cambodia corsica croatia denmark estonia finland france germany latvia lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam"

PERPATCH_OUT="logs/per_patch/s2_b7_native256.csv"
HEADLINE_OUT="logs/polygon_metrics/s2_b7_augmax_full_native256.csv"
rm -f "$PERPATCH_OUT" "$HEADLINE_OUT"

# shellcheck disable=SC2206  # intentional word-splitting
DENSE10_ARGS=( $DENSE10 )
# shellcheck disable=SC2206
FTW22_ARGS=( $FTW22 )

echo "=== S2 B7 per-patch @ native 256 (10 dense held-out) ==="
uv run --no-sync python scripts/eval/per_patch_metrics.py \
    --ckpt "$S2_CKPT" \
    --out "$PERPATCH_OUT" \
    --root data/ftw \
    --dataset-backend s2 \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 256 \
    --countries "${DENSE10_ARGS[@]}"

echo "=== S2 B7 headline polygon metrics @ native 256 (FTW22) ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt "$S2_CKPT" \
    --out "$HEADLINE_OUT" \
    --root data/ftw \
    --dataset-backend s2 \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 256 \
    --countries "${FTW22_ARGS[@]}"

echo "done. wrote:"
echo "  $PERPATCH_OUT"
echo "  $HEADLINE_OUT"
