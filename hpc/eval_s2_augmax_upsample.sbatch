#!/bin/bash
#SBATCH --job-name=ftw-s2aug-up
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/s2aug_up_%j.out
#SBATCH --error=logs/s2aug_up_%j.err

# Eval a retrained augmax S2 model at --upsample-to 512 (resize_factor=2),
# headline polygon metrics over the FTW 22-country test split. Chain after the
# matching training job:
#   sbatch --dependency=afterok:<train> --export=ALL,CKPT_DIR=...,OUT=... hpc/eval_s2_augmax_upsample.sbatch
#
# Required env: CKPT_DIR (run dir to search for the best ckpt), OUT (csv path).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/polygon_metrics

CKPT_DIR="${CKPT_DIR:?need CKPT_DIR}"
OUT="${OUT:?need OUT}"
FTW22="austria belgium brazil cambodia corsica croatia denmark estonia finland france germany latvia lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam"
# shellcheck disable=SC2206
FTW22_ARGS=( $FTW22 )

# best-val/loss checkpoint (save_top_k=1), not last.ckpt.
CKPT="$(find "$CKPT_DIR" -name '*.ckpt' ! -name 'last.ckpt' -print 2>/dev/null | head -1)"
[ -z "$CKPT" ] && CKPT="$(find "$CKPT_DIR" -name 'last.ckpt' -print 2>/dev/null | head -1)"
[ -z "$CKPT" ] && { echo "ERROR: no checkpoint under $CKPT_DIR" >&2; exit 1; }
echo "Using checkpoint: $CKPT -> $OUT"

uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --root data/ftw \
    --dataset-backend s2 --upsample-to 512 \
    --gpu 0 --num-workers 8 \
    --watershed --tta --min-pad-size 512 \
    --countries "${FTW22_ARGS[@]}"

echo "done."
