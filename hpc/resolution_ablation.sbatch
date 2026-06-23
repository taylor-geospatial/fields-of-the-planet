#!/bin/bash
#SBATCH --job-name=ftw-resabl
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/resabl_%j.out
#SBATCH --error=logs/resabl_%j.err

# Resolution ablation: score field polygons against the TRUE FTW polygons under
# three conditions, per country (dense-10), so we can MACRO-average per area bin.
#   planet3m : real 3 m PlanetScope, scored at native 3 m
#   s2up     : Sentinel-2 upsampled to ~2.5 m ("super-resolved", PRUE protocol)
#   s2nat10  : Sentinel-2 capped to its genuine 10 m ground resolution
# The s2up-vs-planet3m gap = value of REAL 3 m over upsampling; s2nat10 = sensor limit.
# Select with: sbatch --export=ALL,CONDITION=planet3m|s2up|s2nat10 resolution_ablation.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONDITION="${CONDITION:?set CONDITION=planet3m|s2up|s2nat10}"
GT="data/ftw_polygons_clipped"
DENSE10="belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden"
COMMON=(--watershed --tta --area-bins 0.5,2 --min-pad-size 512 --gpu 0 --num-workers 8 --gt-polygons-root "$GT")

case "$CONDITION" in
  planet3m)
    CKPT="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    BACKEND=(--root data --dataset-backend planet --pixel-size-m 3)
    ;;
  s2up)
    CKPT="logs/best_checkpoints/s2_efnet7_best.ckpt"
    BACKEND=(--root data/ftw --dataset-backend s2 --upsample-to 512 --pixel-size-m 5)
    ;;
  s2nat10)
    CKPT="logs/best_checkpoints/s2_efnet7_best.ckpt"
    BACKEND=(--root data/ftw --dataset-backend s2 --upsample-to 512 --pixel-size-m 5 --score-gsd-m 10)
    ;;
  *) echo "unknown CONDITION=$CONDITION" >&2; exit 1 ;;
esac

OUTDIR="logs/resolution_ablation/$CONDITION"
mkdir -p "$OUTDIR"
echo "CONDITION=$CONDITION ckpt=$CKPT"
for c in $DENSE10; do
  echo "=== $CONDITION :: $c ==="
  OUT="$OUTDIR/$c.csv"
  rm -f "$OUT" "$OUT.bins.csv"
  uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
      --ckpt "$CKPT" --out "$OUT" "${BACKEND[@]}" "${COMMON[@]}" --countries "$c"
done
echo "done $CONDITION."
