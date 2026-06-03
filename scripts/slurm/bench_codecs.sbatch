#!/bin/bash
#SBATCH --job-name=ftw-codec
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=logs/codec_%j.out
#SBATCH --error=logs/codec_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

LOCAL="${SLURM_TMPDIR:-/tmp}/ftw_codec_$SLURM_JOB_ID"
mkdir -p "$LOCAL"
echo "=== local scratch ==="; df -hT "$LOCAL" | tail -1

echo "=== build 6 full scenes (8192px) in 3 codecs from one read pass ==="
uv run --no-sync python scripts/build_scene.py --out-dir "$LOCAL" --scenes 6 --size 8192

echo
echo "=== full-scene read -> GPU unfold -> infer, 20% overlap, per codec ==="
for fmt in none deflate_pred zstd; do
    uv run --no-sync python scripts/bench_fullscene.py --scene-dir "$LOCAL/$fmt" \
        --overlap 102 --readers 32
done

rm -rf "$LOCAL"
