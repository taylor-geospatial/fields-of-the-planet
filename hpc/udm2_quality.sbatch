#!/bin/bash
#SBATCH --job-name=ftw-udm2
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/udm2_%j.out
#SBATCH --error=logs/udm2_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

UDM2_WORKERS="${UDM2_WORKERS:-32}"

uv run --no-sync scripts/pipeline/udm2_quality.py \
    --planet-root data/planet \
    --country all \
    --workers "$UDM2_WORKERS" \
    --out data/planet/_global/udm2_quality.jsonl
