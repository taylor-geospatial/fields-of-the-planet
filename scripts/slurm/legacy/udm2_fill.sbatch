#!/bin/bash
#SBATCH --job-name=ftw-udmfill
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/udmfill_%j.out
#SBATCH --error=logs/udmfill_%j.err

# Polls for SR tifs lacking a UDM2 partner, activates UDM2 for those scenes,
# range-reads each patch's window. Designed to run in PARALLEL with SR-only
# resample so UDM2 (~3x slower) stays ahead of SR extraction.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

UF_CC="${UF_CC:-64}"
POLL_SEC="${POLL_SEC:-120}"
IDLE_CHECKS="${IDLE_CHECKS:-3}"

uv run --no-sync scripts/udm2_fill.py \
    --planet-root data/planet \
    --concurrency "$UF_CC" \
    --poll-seconds "$POLL_SEC" \
    --idle-checks "$IDLE_CHECKS"
