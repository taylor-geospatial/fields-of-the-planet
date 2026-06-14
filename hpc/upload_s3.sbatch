#!/bin/bash
#SBATCH --job-name=ftw-upload
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/upload_%j.out
#SBATCH --error=logs/upload_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# AWS creds come from the submitting shell via sbatch --export=ALL.
if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" || -z "${AWS_SESSION_TOKEN:-}" ]]; then
    echo "FATAL: AWS_* env vars not set" >&2
    exit 2
fi

PATH="$HOME/.local/bin:$PATH"

# Saturate the network: 64 concurrent in-flight requests, 64 MiB multipart
# chunks. Defaults (10 / 8 MiB) leave bandwidth on the table for ~5 GB tars.
aws configure set default.s3.max_concurrent_requests 64
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 64MB

BUCKET_BASE="s3://us-west-2.opendata.source.coop/ftw/ftw-planet"
INDEX_DST="$BUCKET_BASE/index.parquet"
TARS_DST="$BUCKET_BASE/dataset/"

echo "===== uploading index.parquet ====="
aws s3 cp data/planet/index.parquet "$INDEX_DST" --only-show-errors
echo "  done"

echo "===== syncing data/planet/bulk/ -> $TARS_DST ====="
# `aws s3 sync` already parallelises across files using max_concurrent_requests.
aws s3 sync data/planet/bulk/ "$TARS_DST" \
    --exclude "*" --include "*.tar" \
    --only-show-errors
echo "===== done ====="
aws s3 ls "$BUCKET_BASE/" --recursive --human-readable --summarize | tail -30
