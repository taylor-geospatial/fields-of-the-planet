#!/bin/bash
#SBATCH --job-name=ftw-resexnow
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/resexnow_%j.out
#SBATCH --error=logs/resexnow_%j.err

# Standalone resample extractor — picks up scenes that already have SR URLs
# in _global/resample/sr_activations.jsonl while the main resample_v2 job is
# still activating. Idempotent.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

REX_CC="${REX_CC:-32}"

uv run --no-sync scripts/resample_extract_now.py \
    --planet-root data/planet \
    --concurrency "$REX_CC"
