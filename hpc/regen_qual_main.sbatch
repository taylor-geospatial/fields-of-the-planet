#!/bin/bash
#SBATCH --job-name=ftw-qmain
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/qmain_%j.out
#SBATCH --error=logs/qmain_%j.err

# Regenerate only the main qualitative figure (paper/figs/qualitative_main.pdf).
# Final two columns now vectorize the post-processed field mask into polygons
# (rasterio.features.shapes = GDAL connected-component polygonize, what PQ
# scores) and render with geopandas. Needs a GPU -> gpu partition, not login.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="paper/scripts:${PYTHONPATH:-}"

uv run --no-sync python paper/scripts/qualitative_main.py

echo "done."
