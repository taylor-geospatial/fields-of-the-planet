#!/bin/bash
#SBATCH --job-name=ftw-s2up-eval
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s2up_eval_%j.out
#SBATCH --error=logs/s2up_eval_%j.err

# Resized-S2 control eval. Scores the upsampled-S2 B3 model on the 10 dense
# held-out countries with TTA+watershed at 512 (mirrors the Planet/native-S2
# polygon-metric runs, adding --upsample-to 512). Chain after the training job:
#   sbatch --dependency=afterok:<train_jobid> hpc/eval_s2_upsampled.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/polygon_metrics

CKPT_DIR="logs/prue/ftw_s2-unet-efnet3-crop512-upsampled-v3-augmax-full"
# Match the Table 3 protocol: evaluate the best-val/loss checkpoint (the run
# uses ModelCheckpoint monitor=val/loss, save_top_k=1), NOT last.ckpt. The
# 'val/loss' in the filename creates a subdir, so the file is the lone non-last
# '*.ckpt'. Fall back to last.ckpt only if that is somehow absent.
CKPT="$(find "$CKPT_DIR" -name '*.ckpt' ! -name 'last.ckpt' -print 2>/dev/null | head -1)"
if [ -z "$CKPT" ]; then
    CKPT="$(find "$CKPT_DIR" -name 'last.ckpt' -print 2>/dev/null | head -1)"
fi
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found under $CKPT_DIR" >&2
    exit 1
fi
echo "Using checkpoint: $CKPT"

uv run --no-sync python scripts/eval/polygon_metrics_eval.py \
    --ckpt "$CKPT" \
    --out logs/polygon_metrics/s2_upsampled_b3_augmax_full.csv \
    --root data/ftw \
    --dataset-backend s2 --upsample-to 512 \
    --min-pad-size 512 --watershed --tta \
    --gpu 0 --num-workers 8 \
    --countries belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden

echo "done."
