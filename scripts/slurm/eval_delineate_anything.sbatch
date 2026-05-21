#!/bin/bash
#SBATCH --job-name=ftw-da
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/da_%j.out
#SBATCH --error=logs/da_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

OUT="${OUT:-logs/polygon_metrics/delineate_anything.csv}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.5}"
IMGSZ="${IMGSZ:-1024}"
SPLIT="${SPLIT:-test}"
COUNTRIES="${COUNTRIES:-}"

EXTRA=()
if [[ -n "$COUNTRIES" ]]; then
    # shellcheck disable=SC2206
    EXTRA+=( --countries $COUNTRIES )
fi

mkdir -p "$(dirname "$OUT")"

uv run --no-sync python scripts/eval_delineate_anything.py \
    --out "$OUT" \
    --split "$SPLIT" \
    --conf "$CONF" --iou "$IOU" --imgsz "$IMGSZ" \
    --gpu 0 --num-workers 4 \
    "${EXTRA[@]}"
