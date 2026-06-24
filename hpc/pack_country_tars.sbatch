#!/bin/bash
#SBATCH --job-name=ftw-tar
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tar_%j.out
#SBATCH --error=logs/tar_%j.err

# Pack per-country WebDataset-compatible tar shards into data/planet/bulk/.
# Reads data/planet/index.parquet for the patch list + metadata, and embeds each
# patch's clipped FTW polygons from data/ftw_polygons_clipped/<country>/<pid>.parquet
# (run clip_polygons.sbatch with SPLIT=all FIRST; chain with
# --dependency=afterok:$CLIP_JOBID). Samples whose polygon file is missing are
# dropped and reported as "missing polygons" in the per-country log line.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

TAR_WORKERS="${TAR_WORKERS:-8}"
TAR_COUNTRY="${TAR_COUNTRY:-all}"

uv run --no-sync scripts/pipeline/pack_country_tars.py \
    --planet-root data/planet \
    --workers "$TAR_WORKERS" \
    --country "$TAR_COUNTRY"
