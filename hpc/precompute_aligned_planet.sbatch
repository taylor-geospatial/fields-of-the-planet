#!/bin/bash
#SBATCH --job-name=planet-align
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/planet_align_%j.out
#SBATCH --error=logs/planet_align_%j.err

# Precompute spatially-aligned Planet TIFs (reproject to S2 CRS+bounds at
# native ~3 m GSD, reorder BGR(N)->RGB(N)).  CPU-only; ~41k patches x 2 windows.
# Output: data/planet/aligned_window_a/<country>/<id>.tif (+ window_b)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

PLANET_ROOT="${PLANET_ROOT:-data/planet}"
S2_ROOT="${S2_ROOT:-data/ftw}"
WORKERS="${WORKERS:-108}"   # leave a few CPUs for OS/IO

echo "planet_root: $PLANET_ROOT"
echo "s2_root:     $S2_ROOT"
echo "workers:     $WORKERS"

uv run --no-sync python scripts/pipeline/precompute_aligned_planet.py \
    --planet-root "$PLANET_ROOT" \
    --s2-root "$S2_ROOT" \
    --workers "$WORKERS"
