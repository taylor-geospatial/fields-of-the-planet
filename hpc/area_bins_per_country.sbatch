#!/bin/bash
#SBATCH --job-name=ftw-abpc
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/abpc_%j.out
#SBATCH --error=logs/abpc_%j.err

# Per-country small-field area-bin metrics for the paper's regional delta figure.
#
# Same model pair / protocol as fig:per_country_bars (watershed + TTA,
# --min-pad-size 512, area bins 0.5,2 ha) but run ONCE PER COUNTRY so the pooled
# <out>.bins.csv "small" row is per-country, not pooled over the 23 regions.
# We then harvest the small row from each per-country .bins.csv and combine into
# one CSV per model with rows: country,pq_small,ap_small,n_gt_small.
#
# Pixel size for area->ha: Planet native = 3 m; S2 upsample-512 = 5 m.
# Select model with: sbatch --export=ALL,MODEL=planet|s2_b7 area_bins_per_country.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODEL="${MODEL:?set MODEL=planet or MODEL=s2_b7}"
OUTDIR=logs/area_bins_per_country
# Optional: score against TRUE FTW polygons instead of the rasterized GT mask.
# Set GT_ROOT=data/ftw_polygons_clipped; outputs get a _truegt suffix.
SUFFIX=""
GT_ARGS=()
if [[ -n "${GT_ROOT:-}" ]]; then
  GT_ARGS=(--gt-polygons-root "$GT_ROOT")
  SUFFIX="_truegt"
fi
# Optional: cap predictions to the sensor's native ground resolution before
# matching (e.g. SCORE_GSD=10 for Sentinel-2), so upsampling cannot manufacture
# sub-pixel boundaries. Appends "_gsd<N>" to the suffix.
if [[ -n "${SCORE_GSD:-}" ]]; then
  GT_ARGS+=(--score-gsd-m "$SCORE_GSD")
  SUFFIX="${SUFFIX}_gsd${SCORE_GSD}"
fi
TMPDIR="$OUTDIR/tmp_${MODEL}${SUFFIX}"
mkdir -p "$TMPDIR"

# 23 full-data regions (matches logs/polygon_metrics/*_full_22.csv).
COUNTRIES="austria belgium brazil cambodia corsica croatia denmark estonia finland france germany latvia lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam"

case "$MODEL" in
  planet)
    CKPT="logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt"
    BACKEND=(--root data --dataset-backend planet --pixel-size-m 3)
    FINAL="$OUTDIR/planet_b3_full23_small${SUFFIX}.csv"
    ;;
  s2_b7)
    CKPT="logs/best_checkpoints/s2_efnet7_best.ckpt"
    BACKEND=(--root data/ftw --dataset-backend s2 --upsample-to 512 --pixel-size-m 5)
    FINAL="$OUTDIR/s2_b7_full23_small${SUFFIX}.csv"
    ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

COMMON=(--watershed --tta --area-bins 0.5,2 --min-pad-size 512 --gpu 0 --num-workers 8)
# FINAL already carries ${SUFFIX} (set in the case block); derive the overall
# CSV from its stem so both share the same _truegt[_gsdN] suffix.
FINAL_OVERALL="${FINAL%_small*}_overall${SUFFIX}.csv"

echo "MODEL=$MODEL ckpt=$CKPT final=$FINAL overall=$FINAL_OVERALL"
echo "country,pq_small,ap_small,n_gt_small" > "$FINAL"
echo "country,pq" > "$FINAL_OVERALL"

for c in $COUNTRIES; do
  echo "=== $MODEL :: $c ==="
  OUT="$TMPDIR/$c.csv"
  rm -f "$OUT" "$OUT.bins.csv"
  uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
      --ckpt "$CKPT" --out "$OUT" "${BACKEND[@]}" "${COMMON[@]}" "${GT_ARGS[@]}" --countries "$c"

  # Panel 3: "small" row from the per-country pooled .bins.csv.
  # area_bins field is literally "0.5,2" (embedded comma) -> data rows have 10
  # comma-tokens: $1,$2=area_bins; $3=bin; $4=n_gt; $6=pq; $10=ap_5_95.
  awk -F, -v c="$c" '$3=="small"{printf "%s,%s,%s,%s\n", c, $6, $10, $4}' \
      "$OUT.bins.csv" >> "$FINAL"

  # Panel 1: overall PQ from the per-country main CSV (header + 1 data row;
  # columns: train_checkpoint,country,n_patches,pq_sq,pq_rq,pq,... -> pq=$6).
  awk -F, -v c="$c" 'NR==2{printf "%s,%s\n", c, $6}' "$OUT" >> "$FINAL_OVERALL"
done

echo "done. wrote $FINAL + $FINAL_OVERALL"
cat "$FINAL"; echo; cat "$FINAL_OVERALL"
