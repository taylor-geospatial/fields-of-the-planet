#!/bin/bash
#SBATCH --job-name=ftw-manifest
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/manifest_%j.out
#SBATCH --error=logs/manifest_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

uv run --no-sync scripts/build_manifest.py --ftw-root data/ftw --out data/planet
