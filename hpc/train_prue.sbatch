#!/bin/bash
#SBATCH --job-name=ftw-prue
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# PRUE EfficientNet-B3 UNet on FTW-Planet (PlanetScope SR + FTW v2 labels).
# Single H100, bf16-mixed, batch 64, random 256x256 crops.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CONFIG="${CONFIG:-configs/prue/ftw_planet_efnet3.yaml}"
CKPT_PATH="${CKPT_PATH:-}"
# Multi-seed sweeps: export SEED=<int> before sbatch to override the YAML's
# seed_everything (currently 7) and tag W&B run name + default_root_dir with
# _seed<N> so checkpoints/logs don't collide. Empty SEED keeps YAML default.
SEED="${SEED:-}"

nvidia-smi || true

EXTRA_ARGS=()
if [ -n "$SEED" ]; then
    BASE_NAME="$(basename "$CONFIG" .yaml)"
    RUN_TAG="${BASE_NAME}_seed${SEED}"
    EXTRA_ARGS+=(
        "--seed_everything=${SEED}"
        "--trainer.default_root_dir=logs/prue/${RUN_TAG}"
        "--trainer.logger.init_args.name=${RUN_TAG}"
        "--trainer.logger.init_args.save_dir=logs/prue/${RUN_TAG}"
    )
    echo "Seed override: ${SEED} (run tag: ${RUN_TAG})"
fi

# Note: `--` separates click's CLI_ARGS from ftw's -c/--ckpt_path; LightningCLI
# consumes everything after it. ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} is the
# `set -u`-safe expansion for a possibly-empty bash array.
if [ -n "$CKPT_PATH" ]; then
    echo "Resuming from checkpoint: $CKPT_PATH"
    uv run --no-sync ftw model fit -c "$CONFIG" --ckpt_path "$CKPT_PATH" -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    uv run --no-sync ftw model fit -c "$CONFIG" -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
