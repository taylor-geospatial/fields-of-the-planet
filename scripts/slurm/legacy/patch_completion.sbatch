#!/bin/bash
#SBATCH --job-name=ftw-complete
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/complete_%j.out
#SBATCH --error=logs/complete_%j.err

# Drive missing patches toward 100% SR coverage. For each missing
# (country, id, window) try a prioritised list of recovery strategies:
# original scene -> existing resample candidates -> fresh wider search.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CO_CC="${CO_CC:-64}"

uv run --no-sync scripts/patch_completion.py \
    --planet-root data/planet \
    --concurrency "$CO_CC" \
    --fresh-search \
    --fresh-search-days 1500 \
    --fresh-scene-max-cloud 0.70 \
    --max-fresh-candidates 10
