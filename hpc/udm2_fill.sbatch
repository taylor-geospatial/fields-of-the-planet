#!/bin/bash
#SBATCH --job-name=ftw-udmfv2
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/udmfv2_%j.out
#SBATCH --error=logs/udmfv2_%j.err

# 3-phase UDM2 fill (plan -> activate -> extract). Override:
#   sbatch --export=ALL,PHASE=activate,UF_CC=64 scripts/slurm/udm2_fill_v2.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PHASE="${PHASE:-all}"
UF_CC="${UF_CC:-64}"

uv run --no-sync scripts/udm2_fill_v2.py \
    --planet-root data/planet \
    --phase "$PHASE" \
    --concurrency "$UF_CC"
