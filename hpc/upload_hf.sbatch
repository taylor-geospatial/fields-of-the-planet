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

# Upload data/planet/bulk/*.tar to the HuggingFace dataset repo (one node, off the
# login node). hf_xet is pulled in transiently via `uv run --with` for fast
# large-file transfer + dedup -- no change to the project lockfile. Idempotent:
# shards already on the repo at the same byte size are skipped, so a re-run after
# a timeout resumes. README.md / index.parquet are uploaded separately.
#
# HF_TOKEN comes from the submitting shell via sbatch --export=ALL.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "FATAL: HF_TOKEN not set (submit with --export=ALL after exporting it)" >&2
    exit 2
fi

uv run --no-sync --with hf_xet python scripts/pipeline/upload_hf.py
echo "done."
