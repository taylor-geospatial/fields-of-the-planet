#!/bin/bash
#SBATCH --job-name=ftw-proto
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/proto_%j.out
#SBATCH --error=logs/proto_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

LOCAL="${SLURM_TMPDIR:-/tmp}/ftw_proto_$SLURM_JOB_ID"
mkdir -p "$LOCAL"

echo "=== deps ==="; uv pip install --quiet nvidia-nvcomp-cu12 cupy-cuda12x tifffile

echo "=== build 4 deflate+predictor scenes (8192px) on local NVMe ==="
uv run --no-sync python scripts/build_scene.py --out-dir "$LOCAL" --scenes 4 --size 8192

echo "=== prototype: GPU read path (nvCOMP RAW deflate + predictor undo) ==="
uv run --no-sync python scripts/prototype_gpu_readpath.py --scene-dir "$LOCAL/deflate_pred" \
    --overlap 102 --iters 8

rm -rf "$LOCAL"
