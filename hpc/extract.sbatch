#!/bin/bash
#SBATCH --job-name=ftw-extract
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-63%16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err

# 64 shards across all unique scenes. %16 caps concurrent tasks to keep
# Planet/GCS happy. Each task opens its scenes' COGs once and range-reads
# every member patch.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

NUM_SHARDS=64
EC="${EC:-16}"

uv run --no-sync scripts/pipeline/extract_shard.py \
    --out data/planet \
    --shard-id "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS" \
    --concurrency "$EC"
