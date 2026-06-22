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
BACKEND="${BACKEND:-planet}"
ROOT="${ROOT:-data}"
PIXSIZE="${PIXSIZE:-3}"
AREABINS="${AREABINS:-0.5,2}"
CEILING="${CEILING:-3000}"
HFREPO="${HFREPO:-torchgeo/delineate-anything}"
HFFILE="${HFFILE:-delineate_anything_rgb_yolo11x-88ede029.pt}"
CONF="${CONF:-0.005}"
IOU="${IOU:-0.5}"
IMGSZ="${IMGSZ:-512}"
SPLIT="${SPLIT:-test}"
COUNTRIES="${COUNTRIES:-}"
SAMPLE="${SAMPLE:-}"

EXTRA=()
if [[ -n "$COUNTRIES" ]]; then
    # shellcheck disable=SC2206
    EXTRA+=( --countries $COUNTRIES )
fi
if [[ -n "$SAMPLE" ]]; then
    EXTRA+=( --save-rgb-sample "$SAMPLE" )
fi

mkdir -p "$(dirname "$OUT")"

uv run --no-sync --with ultralytics python scripts/eval/eval_delineate_anything.py \
    --out "$OUT" \
    --dataset-backend "$BACKEND" --root "$ROOT" \
    --area-bins "$AREABINS" --pixel-size-m "$PIXSIZE" --rgb-ceiling "$CEILING" \
    --hf-repo "$HFREPO" --hf-file "$HFFILE" \
    --split "$SPLIT" \
    --conf "$CONF" --iou "$IOU" --imgsz "$IMGSZ" \
    --gpu 0 --num-workers 4 \
    "${EXTRA[@]}"
