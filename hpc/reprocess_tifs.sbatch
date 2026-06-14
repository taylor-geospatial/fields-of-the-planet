#!/bin/bash
#SBATCH --job-name=ftw-reproc
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reproc_%j.out
#SBATCH --error=logs/reproc_%j.err

# Walk every GeoTIFF under data/planet/<country>/ and rewrite it in place
# with ZSTD-22, single-strip layout, no overviews, and provenance tags
# embedded from the pipeline's JSONL logs. Idempotent (skips files with
# PROCESSING_VERSION already set).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

RP_WORKERS="${RP_WORKERS:-64}"
RP_COUNTRY="${RP_COUNTRY:-all}"

uv run --no-sync scripts/pipeline/reprocess_tifs.py \
    --planet-root data/planet \
    --country "$RP_COUNTRY" \
    --workers "$RP_WORKERS"
