#!/bin/bash
#SBATCH --job-name=ftw-improvefig
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/improvefig_%j.out
#SBATCH --error=logs/improvefig_%j.err

# Renders paper/figs/improvement_examples.pdf (+ logs/improvement_examples.png)
# from the per-patch CSVs. Re-runs both models (TTA+watershed) for the selected
# top patches, so it needs a GPU.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TOP_N="${TOP_N:-5}"
MIN_N_GT="${MIN_N_GT:-8}"

# tg_style lives in paper/scripts; put it on the path.
PYTHONPATH="paper/scripts:${PYTHONPATH:-}" uv run --no-sync python paper/scripts/improvement_figure.py \
    --planet-csv logs/per_patch/planet_b3.csv \
    --s2-csv logs/per_patch/s2_b7.csv \
    --top-n "$TOP_N" \
    --min-n-gt "$MIN_N_GT" \
    --out paper/figs/improvement_examples.pdf \
    --png logs/improvement_examples.png

echo "done."
