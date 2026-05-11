#!/bin/bash
#SBATCH --job-name=ftw-restore
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/restore_%j.out
#SBATCH --error=logs/restore_%j.err

# Re-extract SR for patches that lost their original tif during a failed
# resample re-extract pass. Uses the ORIGINAL item_id from extract shards.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

RS_CC="${RS_CC:-64}"

uv run --no-sync scripts/restore_lost_sr.py \
    --planet-root data/planet \
    --concurrency "$RS_CC"
