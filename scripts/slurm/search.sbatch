#!/bin/bash
#SBATCH --job-name=ftw-search
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-31%8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/search_%A_%a.out
#SBATCH --error=logs/search_%A_%a.err

# 32 shards of the global manifest. %8 caps concurrent tasks at 8 to stay
# polite to the Data API. Each task is search-only — cheap metadata.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

NUM_SHARDS=32
MAX_CC="${MAX_CC:-0.1}"
SC="${SC:-16}"

uv run --no-sync scripts/search_shard.py \
    --out data/planet \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --concurrency "$SC" \
    --max-cloud-cover "$MAX_CC"
