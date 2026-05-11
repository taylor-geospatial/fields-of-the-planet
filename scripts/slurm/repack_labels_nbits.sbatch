#!/bin/bash
#SBATCH --job-name=ftw-relbl
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/relbl_%j.out
#SBATCH --error=logs/relbl_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

RL_WORKERS="${RL_WORKERS:-64}"
RL_COUNTRY="${RL_COUNTRY:-all}"

uv run --no-sync scripts/repack_labels_nbits.py \
    --planet-root data/planet \
    --workers "$RL_WORKERS" \
    --country "$RL_COUNTRY"
