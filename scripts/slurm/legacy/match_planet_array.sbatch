#!/bin/bash
#SBATCH --job-name=ftw-match
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-24%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/match_%A_%a.out
#SBATCH --error=logs/match_%A_%a.err

# One SLURM array task per FTW v2 country (25 total).
# %6 caps concurrent tasks to 6 to stay polite with Planet's API.
# Override knobs at submit time, e.g.:
#   sbatch --export=ALL,MAX_CC=0.1,SC=64,AC=64,EC=32 scripts/slurm/match_planet_array.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

COUNTRIES=(austria belgium brazil cambodia corsica croatia denmark estonia finland france
           germany india kenya latvia lithuania luxembourg netherlands portugal rwanda
           slovakia slovenia south_africa spain sweden vietnam)
COUNTRY="${COUNTRIES[$SLURM_ARRAY_TASK_ID]}"

MAX_CC="${MAX_CC:-0.1}"
SC="${SC:-64}"   # search concurrency
AC="${AC:-64}"   # activate concurrency
EC="${EC:-32}"   # extract (per-scene) concurrency

mkdir -p logs

echo "[$(date)] starting match for $COUNTRY  (cc=$MAX_CC sc=$SC ac=$AC ec=$EC)"
uv run --no-sync scripts/match_planet.py \
    --country "$COUNTRY" \
    --ftw-root data/ftw \
    --out data/planet \
    --max-cloud-cover "$MAX_CC" \
    --search-concurrency "$SC" \
    --activate-concurrency "$AC" \
    --extract-concurrency "$EC"
echo "[$(date)] done with $COUNTRY"
