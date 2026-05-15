#!/bin/bash
#SBATCH --job-name=ftw-index
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/index_%j.out
#SBATCH --error=logs/index_%j.err

# Build per-country index.parquet over the migrated FTW-aligned layout.
# Reads TIFF tags + joins with _global/manifest.jsonl for geometry.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

IDX_WORKERS="${IDX_WORKERS:-16}"
IDX_COUNTRY="${IDX_COUNTRY:-all}"

uv run --no-sync scripts/build_country_indexes.py \
    --planet-root data/planet \
    --workers "$IDX_WORKERS" \
    --country "$IDX_COUNTRY"
