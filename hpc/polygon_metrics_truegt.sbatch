#!/bin/bash
#SBATCH --job-name=ftw-truegt
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/truegt_%j.out
#SBATCH --error=logs/truegt_%j.err

# Polygon metrics for the Planet model scored two ways, identical flags otherwise,
# so they are directly comparable:
#   (1) rasterized-GT : connected components of the rasterized GT mask (status quo)
#   (2) true-GT       : the actual FTW vector parcels (data/ftw_polygons_clipped)
# Run both here rather than diffing against an older CSV, to rule out any flag drift.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/polygon_metrics

CKPT="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
FTW22="austria belgium brazil cambodia corsica croatia denmark estonia finland france germany latvia lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam"
COMMON=(--ckpt "$CKPT" --root data --dataset-backend planet --min-pad-size 512 \
        --watershed --tta --gpu 0 --num-workers 8)
# shellcheck disable=SC2206
C22=( $FTW22 )

RASTER_OUT="logs/polygon_metrics/planet_b3_rastergt_22.csv"
TRUE_OUT="logs/polygon_metrics/planet_b3_truegt_22.csv"
rm -f "$RASTER_OUT" "$TRUE_OUT"

echo "=== (1) rasterized-mask GT ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    "${COMMON[@]}" --out "$RASTER_OUT" --countries "${C22[@]}"

echo "=== (2) true FTW polygon GT ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    "${COMMON[@]}" --gt-polygons-root data/ftw_polygons_clipped \
    --out "$TRUE_OUT" --countries "${C22[@]}"

echo "done. wrote:"
echo "  $RASTER_OUT"
echo "  $TRUE_OUT"
