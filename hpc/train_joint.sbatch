#!/bin/bash
#SBATCH --job-name=ftw-joint
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=logs/joint_%j.out
#SBATCH --error=logs/joint_%j.err

# Joint Planet (3m) + Sentinel-2 (10m) training.
# FTWJointDataModule: 50/50 mixed batches, S2 bilinear-upsampled 256→512,
# per-modality normalisation in dataset wrappers.
# Single H100 (--constraint=h100), bf16-mixed, batch 32, 512x512 crops.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Rasterio / GDAL read-speed knobs
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_CACHEMAX=512
export GDAL_NUM_THREADS=2
export OMP_NUM_THREADS=1

CONFIG="${CONFIG:-configs/prue/ftw_joint_efnet3_crop512_v3_augmax_full.yaml}"
CKPT_PATH="${CKPT_PATH:-}"
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

if [ -n "$CKPT_PATH" ]; then
    echo "Resuming from: $CKPT_PATH"
    uv run --no-sync ftw model fit -c "$CONFIG" --ckpt_path "$CKPT_PATH" -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    uv run --no-sync ftw model fit -c "$CONFIG" -- ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi
