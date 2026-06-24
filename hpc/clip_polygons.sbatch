#!/bin/bash
#SBATCH --job-name=ftw-clippoly
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/clippoly_%j.out
#SBATCH --error=logs/clippoly_%j.err

# Clip the original FTW field-boundary polygons to per-patch bounds, writing one
# GeoParquet per patch under data/ftw_polygons_clipped/<country>/<pid>.parquet.
# These are the TRUE field geometries (reprojected to each patch's Planet UTM
# grid, clipped to bounds) for scoring polygon metrics against real polygons
# instead of connected components of the rasterized GT mask, and for embedding
# in the WebDataset tars (pack_country_tars.py). CPU/IO only.
# Default SPLIT=all covers train/val/test (the full dataset, ~66k patches);
# override with SPLIT=test for the eval subset only.
# Idempotent/resumable: existing per-patch files are skipped (use --overwrite to
# force). Override the country set with: sbatch --export=ALL,COUNTRIES="germany latvia" ...

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SPLIT="${SPLIT:-all}"

EXTRA=()
if [[ -n "${COUNTRIES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=(--countries ${COUNTRIES})
fi

uv run --no-sync python scripts/pipeline/clip_polygons_per_patch.py \
    --split "$SPLIT" "${EXTRA[@]}"

echo "done."
