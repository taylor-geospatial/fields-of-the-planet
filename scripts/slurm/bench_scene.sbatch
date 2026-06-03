#!/bin/bash
#SBATCH --job-name=ftw-scene
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scene_%j.out
#SBATCH --error=logs/scene_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# node-local storage closest to compute (NVMe), NOT the NFS /u mount
LOCAL="${SLURM_TMPDIR:-/tmp}/ftw_scenes_$SLURM_JOB_ID"
mkdir -p "$LOCAL"
echo "=== local scratch: $LOCAL ==="
df -hT "$LOCAL" | tail -1

echo "=== build full scenes (8192px, 8-band zstd) on local NVMe from real tiles ==="
uv run --no-sync python scripts/build_scene.py --out-dir "$LOCAL" --scenes 6 --size 8192

echo
echo "=== end-to-end windowed inference, 20% overlap, reading from local NVMe ==="
uv run --no-sync python scripts/bench_scene.py --scene-dir "$LOCAL" --overlap 102 \
    --workers 16 24 32 48 64

rm -rf "$LOCAL"
