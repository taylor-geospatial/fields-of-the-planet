#!/bin/bash
#SBATCH --job-name=ftw-react
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reactivate_%j.out
#SBATCH --error=logs/reactivate_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

REACT_CC="${REACT_CC:-32}"

uv run --no-sync scripts/reactivate_failed.py \
    --out data/planet \
    --concurrency "$REACT_CC"
