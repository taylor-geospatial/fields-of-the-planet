#!/bin/bash
#SBATCH --job-name=ftw-areabin
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/areabin_%j.out
#SBATCH --error=logs/areabin_%j.err

# Polygon metrics broken down by GT field area (small<0.5 / medium 0.5-2 /
# large>2 ha) AND across IoU thresholds (PQ, SQ, RQ@.5, F1@.75, AP[.5:.95]),
# pooled (micro) over the 10 dense held-out countries. Each run writes a
# per-country CSV + a <out>.bins.csv with the per-bin breakdown.
#
# Pixel size for area->ha: Planet native = 3 m; S2 upsample-512 = 5 m
# (256px@10m chip -> 512px -> 2560m/512 = 5 m).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/area_bins

DENSE10="belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden"
S2_B3_CKPT="logs/prue/ftw_s2-unet-efnet3-crop256-s2-v3-augmax-full/ftw-s2/da62pwt1/checkpoints/epoch=42-val/loss=0.02.ckpt"
COMMON=(--watershed --tta --area-bins 0.5,2 --gpu 0 --num-workers 8)
# shellcheck disable=SC2206
D10=( $DENSE10 )

echo "=== Planet B3 (FTP main) ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt \
    --out logs/area_bins/planet_b3.csv --root data --dataset-backend planet \
    --min-pad-size 512 --pixel-size-m 3 "${COMMON[@]}" --countries "${D10[@]}"

echo "=== S2 B7 (upsample-512) ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt logs/best_checkpoints/s2_efnet7_best.ckpt \
    --out logs/area_bins/s2_b7.csv --root data/ftw --dataset-backend s2 \
    --upsample-to 512 --min-pad-size 512 --pixel-size-m 5 "${COMMON[@]}" --countries "${D10[@]}"

echo "=== S2 B3 (upsample-512) ==="
uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt "$S2_B3_CKPT" \
    --out logs/area_bins/s2_b3.csv --root data/ftw --dataset-backend s2 \
    --upsample-to 512 --min-pad-size 512 --pixel-size-m 5 "${COMMON[@]}" --countries "${D10[@]}"

echo "done."
