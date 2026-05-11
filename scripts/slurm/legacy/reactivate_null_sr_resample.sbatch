#!/bin/bash
#SBATCH --job-name=ftw-resnull
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/resnull_%j.out
#SBATCH --error=logs/resnull_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

RN_CC="${RN_CC:-64}"

uv run --no-sync scripts/reactivate_null_sr_resample.py \
    --out data/planet \
    --concurrency "$RN_CC"
