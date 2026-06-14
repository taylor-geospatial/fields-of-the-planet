#!/bin/bash
#SBATCH --job-name=ftw-rast
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-24%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rast_%A_%a.out
#SBATCH --error=logs/rast_%A_%a.err

# One SLURM task per FTW v2 country. Rasterizes field-boundary polygons
# (3-class: 0=bg, 1=field interior, 2=boundary) onto every Planet SR tif's
# native UTM grid. Skips already-existing label tifs, so safe to re-run as
# extract continues to produce new SR tifs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

COUNTRIES=(austria belgium brazil cambodia corsica croatia denmark estonia finland france
           germany india kenya latvia lithuania luxembourg netherlands portugal rwanda
           slovakia slovenia south_africa spain sweden vietnam)
COUNTRY="${COUNTRIES[$SLURM_ARRAY_TASK_ID]}"

BUFFER_M="${BUFFER_M:-3.0}"
RAST_WORKERS="${RAST_WORKERS:-32}"

uv run --no-sync scripts/pipeline/rasterize_labels.py \
    --country "$COUNTRY" \
    --planet-root data/planet \
    --polygons-root data/ftw_polygons \
    --workers "$RAST_WORKERS" \
    --boundary-buffer-m "$BUFFER_M" \
    --summary-out data/planet/_global/rasterize_summary.jsonl
