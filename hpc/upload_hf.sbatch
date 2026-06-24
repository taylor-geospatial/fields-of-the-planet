#!/bin/bash
#SBATCH --job-name=ftw-hf
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hf_%j.out
#SBATCH --error=logs/hf_%j.err

# Upload data/planet/bulk/*.tar to the HuggingFace dataset repo, off the login
# node. hf_xet is pulled in transiently via `uv run --with` -- no change to the
# project lockfile. Idempotent: shards already on the repo at the same byte size
# are skipped, so a re-run after a timeout resumes. README.md / index.parquet are
# uploaded separately.
#
# Multi-node: a single sequential LFS stream is slow (~NIC-underutilising), so fan
# out one job per node, each uploading a disjoint largest-first slice from its own
# NIC. Submit node-pinned jobs (cpu_amd nodes are partly reserved, so -w + low
# cpus places immediately where --array packing / --exclusive will not):
#   nodes=(rails08 rails09 rails10 rails11 rails14); n=${#nodes[@]}
#   for i in "${!nodes[@]}"; do
#     sbatch -w "${nodes[$i]}" --export=ALL,SLICE_ID=$i,SLICE_COUNT=$n hpc/upload_hf.sbatch
#   done
# Concurrent commits to main may 412; the script retries the commit step (the LFS
# blob is already uploaded by then). Default SLICE_ID=0 SLICE_COUNT=1 = all shards.
#
# HF_TOKEN comes from the submitting shell via sbatch --export=ALL.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "FATAL: HF_TOKEN not set (submit with --export=ALL after exporting it)" >&2
    exit 2
fi

SLICE_ID="${SLICE_ID:-0}"
SLICE_COUNT="${SLICE_COUNT:-1}"

uv run --no-sync --with hf_xet python scripts/pipeline/upload_hf.py \
    --slice-id "$SLICE_ID" --slice-count "$SLICE_COUNT"
echo "done."
