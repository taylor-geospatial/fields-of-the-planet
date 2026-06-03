#!/bin/bash
#SBATCH --job-name=ftw-opt
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH --output=logs/opt_%j.out
#SBATCH --error=logs/opt_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
echo "=== deps ==="; uv pip install --quiet nvidia-nvcomp-cu12 cupy-cuda12x
echo "=== pre-tiled paths: deflate(eager/compiled) vs uncompressed(compiled) ==="
uv run --no-sync python scripts/bench_pretiled.py --tiles 512 --iters 6
