#!/bin/bash
#SBATCH --job-name=ftw-e2e
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=logs/e2e_%j.out
#SBATCH --error=logs/e2e_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# 48 CPU / 1 GPU mirrors the per-GPU CPU ratio of an 8-GPU cloud node
# (p6-b200 = 192 vCPU / 8 GPU = 24/GPU; we give 48 to find the loader ceiling).
uv run --no-sync python scripts/bench_e2e.py --workers 8 16 24 32 48
