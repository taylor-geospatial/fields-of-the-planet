#!/bin/bash
#SBATCH --job-name=ftw-bench
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Cap batch at 128: batch>=256 at 512x512 overflows torch's 32-bit conv
# indexing (>2^31 elems in a full-res decoder tensor). Run as separate
# invocations so a crash in one mode can't suppress the other's results.
BATCHES="8 16 32 64 96 128"

echo "=== bf16, channels_last (eager) ==="
uv run --no-sync python scripts/bench_h100.py --batches $BATCHES

echo
echo "=== bf16, channels_last + torch.compile ==="
uv run --no-sync python scripts/bench_h100.py --batches $BATCHES --compile
