#!/bin/bash
#SBATCH --job-name=ftw-quald
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/quald_%j.out
#SBATCH --error=logs/quald_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

uv run --no-sync python paper/scripts/make_qualitative_delineate.py
