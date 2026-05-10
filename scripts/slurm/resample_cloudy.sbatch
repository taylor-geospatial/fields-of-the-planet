#!/bin/bash
#SBATCH --job-name=ftw-resample
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/resample_%j.out
#SBATCH --error=logs/resample_%j.err

# Per-patch re-search for an alternative PSScene with low cloud cover OVER
# THIS PATCH SPECIFICALLY (not just scene-wide). Probes UDM2 for each
# candidate scene to confirm patch-level quality before activating SR.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MIN_CLEAR="${MIN_CLEAR:-0.95}"
MAX_UNUSABLE="${MAX_UNUSABLE:-0.05}"
SCENE_MAX_CLOUD="${SCENE_MAX_CLOUD:-0.40}"
MAX_CANDS="${MAX_CANDS:-5}"
RES_CC="${RES_CC:-16}"

uv run --no-sync scripts/resample_cloudy.py \
    --planet-root data/planet \
    --min-clear "$MIN_CLEAR" \
    --max-unusable "$MAX_UNUSABLE" \
    --scene-max-cloud "$SCENE_MAX_CLOUD" \
    --max-candidates "$MAX_CANDS" \
    --concurrency "$RES_CC"
