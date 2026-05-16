#!/bin/bash
#SBATCH --job-name=ftw-pp
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/pp_%j.out
#SBATCH --error=logs/pp_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CKPT="${CKPT:?need CKPT}"
OUT="${OUT:?need OUT}"
# shellcheck disable=SC2206  # intentional word-splitting of EXTRA flags
EXTRA_ARGS=( ${EXTRA:---watershed} )

uv run --no-sync python scripts/postprocess_eval.py \
    --ckpt "$CKPT" --out "$OUT" --gpu 0 --num-workers 8 "${EXTRA_ARGS[@]}"
