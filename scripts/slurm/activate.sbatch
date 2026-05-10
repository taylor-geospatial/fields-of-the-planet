#!/bin/bash
#SBATCH --job-name=ftw-activate
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/activate_%j.out
#SBATCH --error=logs/activate_%j.err

# Single global activation pass. Resumable — re-run to continue.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

AC="${AC:-96}"

uv run --no-sync scripts/activate_global.py \
    --out data/planet \
    --concurrency "$AC"
