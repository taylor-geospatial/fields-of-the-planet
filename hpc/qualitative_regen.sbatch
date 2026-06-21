#!/bin/bash
#SBATCH --job-name=ftw-qualregen
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/qualregen_%j.out
#SBATCH --error=logs/qualregen_%j.err

# Regenerates the three qualitative figures with the corrected S2 eval protocol:
#   paper/figs/qualitative_v6_appx.pdf  (raw argmax masks)
#   paper/figs/qualitative_v7_appx.pdf  (TTA + watershed instances)
#   paper/figs/qualitative_v8_main.pdf  (raw + post-proc, main fig)
#
# S2 predictions now use resize_factor=2: the stacked 8-band 256 input is
# bilinear-upsampled to 512 before inference, and the 512 prediction is mapped
# onto the Planet grid via a transform scaled by 256/512 (matches the headline
# tables and make_germany_diagnostic.py). Planet path is unchanged (native 512).
# Needs a GPU -> gpu partition, never the login node.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# tg_style lives in paper/scripts; put it on the path.
export PYTHONPATH="paper/scripts:${PYTHONPATH:-}"

uv run --no-sync python paper/scripts/make_qualitative_v6.py
uv run --no-sync python paper/scripts/make_qualitative_v7.py
uv run --no-sync python paper/scripts/make_qualitative_v8.py

echo "done."
