#!/bin/bash
#SBATCH --job-name=ftw-resa2
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/resav2_%j.out
#SBATCH --error=logs/resav2_%j.err

# 3-phase resample (search -> probe -> extract). Override phase + concurrency:
#   sbatch --export=ALL,PHASE=probe,CC=64 scripts/slurm/resample_v2.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PHASE="${PHASE:-all}"
RES_CC="${RES_CC:-64}"
MAX_CANDS="${MAX_CANDS:-2}"

uv run --no-sync scripts/resample_v2.py \
    --planet-root data/planet \
    --phase "$PHASE" \
    --concurrency "$RES_CC" \
    --max-candidates "$MAX_CANDS"
