#!/bin/bash
#SBATCH --job-name=ftw-germdiag
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/germdiag_%j.out
#SBATCH --error=logs/germdiag_%j.err

# Renders paper/figs/germany_diagnostic.pdf (+ logs/germany_diagnostic.png):
# the worst Germany held-out patches (n_gt>=10, most negative planet-vs-S2
# obj F1) showing Planet's parcel merging vs the S2 baseline. Re-runs both
# models (TTA+watershed) on the selected patches, so it needs a GPU.
#
# S2 panel uses the corrected upsample-512 protocol (s2_b7_upsampled512.csv);
# Planet runs at native 512 (planet_b3.csv).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TOP_N="${TOP_N:-6}"
MIN_N_GT="${MIN_N_GT:-10}"

# tg_style lives in paper/scripts; put it on the path.
PYTHONPATH="paper/scripts:${PYTHONPATH:-}" uv run --no-sync python paper/scripts/make_germany_diagnostic.py \
    --planet-csv logs/per_patch/planet_b3.csv \
    --s2-csv logs/per_patch/s2_b7_upsampled512.csv \
    --country germany \
    --top-n "$TOP_N" \
    --min-n-gt "$MIN_N_GT" \
    --out paper/figs/germany_diagnostic.pdf \
    --png logs/germany_diagnostic.png

echo "done."
