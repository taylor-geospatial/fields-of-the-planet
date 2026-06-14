#!/bin/bash
#SBATCH --job-name=ftw-dl
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ftwdl_%j.out
#SBATCH --error=logs/ftwdl_%j.err

# Pull one FTW country and build the JSONL index.
# Override at submit time:
#   sbatch --export=ALL,COUNTRY=rwanda scripts/slurm/download_ftw.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

COUNTRY="${COUNTRY:-rwanda}"

mkdir -p logs

uv run scripts/pipeline/download_ftw.py --country "$COUNTRY" --root data/ftw
