#!/bin/bash
#SBATCH --job-name=ftw-migrate
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/migrate_%j.out
#SBATCH --error=logs/migrate_%j.err

# Reorganize data/planet/<country>/ from the flat <id>_<a|b|label>.tif
# layout into FTW-aligned window_a/ window_b/ labels/ subdirs. Pure file
# moves (os.replace, atomic within filesystem). Dry-run unless MIG_APPLY=true.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MIG_APPLY="${MIG_APPLY:-false}"

APPLY_FLAG=""
if [[ "$MIG_APPLY" == "true" ]]; then
    APPLY_FLAG="--apply"
fi

uv run --no-sync scripts/pipeline/migrate_to_ftw_layout.py \
    --planet-root data/planet \
    $APPLY_FLAG
