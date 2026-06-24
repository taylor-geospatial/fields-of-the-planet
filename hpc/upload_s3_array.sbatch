#!/bin/bash
#SBATCH --job-name=ftw-upload
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/upload_%A_%a.out
#SBATCH --error=logs/upload_%A_%a.err

# Multi-node tar upload: each task pushes a DISJOINT slice of data/planet/bulk/*.tar
# from its own NIC, so aggregate bandwidth scales with the task count instead of
# bottlenecking on a single node's NIC (the login node must never be used for this).
# Tars are assigned largest-first round-robin so the few multi-GB shards spread out.
#
# Two ways to fan out across distinct NICs:
#   1. Array:  sbatch --array=0-4 ...                    (slice = SLURM_ARRAY_TASK_*)
#      Simple, but SLURM may pack several array tasks onto one fat node (shared NIC),
#      and --exclusive / high cpus-per-task often won't place when cpu_amd nodes are
#      partly reserved for higher-priority partitions.
#   2. Node-pinned (robust NIC spread): submit one job per node, e.g.
#        nodes=(rails08 rails09 rails10 rails11 rails14); n=${#nodes[@]}
#        for i in "${!nodes[@]}"; do
#          sbatch -w "${nodes[$i]}" --export=ALL,SLICE_ID=$i,SLICE_COUNT=$n upload_s3_array.sbatch
#        done
#      Guarantees N distinct nodes/NICs and schedules at low cpus-per-task.
#
# Uploads via `aws s3 cp` (not sync) so every shard is overwritten unconditionally
# -- the local tars are the source of truth. The small README.md / index.parquet
# are uploaded separately (upload_s3.sbatch or by hand); this job is tars only.
#
# AWS creds come from the submitting shell via sbatch --export=ALL.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" || -z "${AWS_SESSION_TOKEN:-}" ]]; then
    echo "FATAL: AWS_* env vars not set (submit with --export=ALL after exporting creds)" >&2
    exit 2
fi

PATH="$HOME/.local/bin:$PATH"
BUCKET_BASE="s3://us-west-2.opendata.source.coop/ftw/ftw-planet"

# Saturate this node's NIC: 64 concurrent in-flight requests, 64 MiB chunks.
aws configure set default.s3.max_concurrent_requests 64
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 64MB

# Slice index: SLURM array vars by default, overridable via SLICE_ID / SLICE_COUNT
# env for node-pinned single jobs (sbatch -w <node> --export=ALL,SLICE_ID=i,SLICE_COUNT=n).
NTASKS="${SLICE_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
TASKID="${SLICE_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

# Largest-first so big shards spread across nodes instead of piling on one.
mapfile -t TARS < <(ls -S data/planet/bulk/*.tar)

echo "[task $TASKID/$NTASKS] host=$(hostname) candidates=${#TARS[@]}"
n=0
for i in "${!TARS[@]}"; do
    if (( i % NTASKS == TASKID )); then
        src="${TARS[$i]}"
        echo "[task $TASKID] uploading $(basename "$src") ..."
        aws s3 cp "$src" "$BUCKET_BASE/dataset/$(basename "$src")" --only-show-errors
        n=$((n + 1))
    fi
done
echo "[task $TASKID] done — uploaded $n shards"
