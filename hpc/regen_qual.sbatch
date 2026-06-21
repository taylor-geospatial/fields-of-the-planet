#!/bin/bash
#SBATCH --job-name=ftw-qregen
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/qregen_%j.out
#SBATCH --error=logs/qregen_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

uv run --no-sync python paper/scripts/qualitative_main.py
uv run --no-sync python paper/scripts/qualitative_raw_appendix.py
uv run --no-sync python paper/scripts/qualitative_instances_appendix.py
# delineate uses the Delineate-Anything YOLO baseline; ultralytics is an
# optional paper-only dep, supplied ephemerally rather than added to the project.
uv run --no-sync --with ultralytics python paper/scripts/qualitative_delineate.py
