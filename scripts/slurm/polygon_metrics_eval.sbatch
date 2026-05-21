#!/bin/bash
#SBATCH --job-name=ftw-polyeval
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/polyeval_%j.out
#SBATCH --error=logs/polyeval_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CKPT="${CKPT:?need CKPT}"
OUT="${OUT:?need OUT}"
BACKEND="${BACKEND:-planet}"
COUNTRIES="${COUNTRIES:?need COUNTRIES (space-separated)}"
# Default root depends on backend: data/planet for planet, data/ftw for s2.
if [ "$BACKEND" = "s2" ]; then
    ROOT="${ROOT:-data/ftw}"
else
    ROOT="${ROOT:-data}"
fi
# shellcheck disable=SC2206  # intentional word-splitting of EXTRA flags
EXTRA_ARGS=( ${EXTRA:---watershed --tta --min-pad-size 512} )
# shellcheck disable=SC2206
COUNTRY_ARGS=( $COUNTRIES )

uv run --no-sync python scripts/polygon_metrics_eval.py \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --root "$ROOT" \
    --dataset-backend "$BACKEND" \
    --gpu 0 --num-workers 8 \
    --countries "${COUNTRY_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
