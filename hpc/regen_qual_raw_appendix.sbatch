#!/bin/bash
#SBATCH --job-name=ftw-qv6
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/qv6_%j.out
#SBATCH --error=logs/qv6_%j.err

# Regenerate only the v6 raw-prediction appendix figure after re-curating its
# rows to seven dense per-held-out-country Planet-wins (delta-filtered from the
# per-patch metrics, disjoint from the main figure and v7). v7/v8 are left
# untouched so the already-verified figures don't change.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="paper/scripts:${PYTHONPATH:-}"

uv run --no-sync python paper/scripts/qualitative_raw_appendix.py
echo "done."
