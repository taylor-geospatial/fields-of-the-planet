#!/bin/bash
#SBATCH --job-name=ftw-match
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/match_%j.out
#SBATCH --error=logs/match_%j.err

# Network-bound (Planet Data + Orders API); cheapest CPU partition (0.4 SU/hr/core).
# Override defaults at submit time, e.g.:
#   sbatch --export=ALL,COUNTRY=rwanda,CONCURRENCY=8 scripts/slurm/match_planet.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

COUNTRY="${COUNTRY:-rwanda}"
CONCURRENCY="${CONCURRENCY:-8}"
SEARCH_DAYS="${SEARCH_DAYS:-14}"
MAX_CC="${MAX_CC:-0.1}"
RES="${RES:-3.0}"

mkdir -p logs

# .env (PL_API_KEY) loaded by the script via python-dotenv.
uv run scripts/match_planet.py \
    --country "$COUNTRY" \
    --ftw-root data/ftw \
    --out data/planet \
    --search-days "$SEARCH_DAYS" \
    --max-cloud-cover "$MAX_CC" \
    --resolution "$RES" \
    --concurrency "$CONCURRENCY"
