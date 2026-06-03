#!/bin/bash
#SBATCH --job-name=ftw-gpudec
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gpudec_%j.out
#SBATCH --error=logs/gpudec_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=== install nvcomp + cupy (cuda12) into venv ==="
uv pip install --quiet nvidia-nvcomp-cu12 cupy-cuda12x

echo "=== GPU decode throughput: Deflate vs GDeflate ==="
uv run --no-sync python scripts/bench_gpu_decode.py --tiles 512 --iters 20
