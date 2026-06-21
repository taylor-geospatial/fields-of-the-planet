#!/bin/bash
#SBATCH --job-name=ftw-ppup
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ppup_%j.out
#SBATCH --error=logs/ppup_%j.err

# tab:heldout at upsample-512: S2 B3 + B7 postprocessing ablation (4 WS x TTA
# combos) re-run with --upsample-to 512 (resize_factor=2), to fill the last
# pre-upsample gap. Writes logs/postproc_ablation/<stem>_upsampled_<combo>.csv.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/postproc_ablation
DENSE10="belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden"
# shellcheck disable=SC2206
D10=( $DENSE10 )
S2_B3="logs/prue/ftw_s2-unet-efnet3-crop256-s2-v3-augmax-full/ftw-s2/da62pwt1/checkpoints/epoch=42-val/loss=0.02.ckpt"
S2_B7="logs/best_checkpoints/s2_efnet7_best.ckpt"

run() {  # $1=ckpt $2=stem $3=combo $4..=extra flags
    local ckpt="$1" stem="$2" combo="$3"; shift 3
    echo "=== $stem $combo ==="
    uv run --no-sync python scripts/eval/postprocess_eval.py \
        --ckpt "$ckpt" --out "logs/postproc_ablation/${stem}_upsampled_${combo}.csv" \
        --root data/ftw --dataset-backend s2 --upsample-to 512 --min-pad-size 512 \
        --gpu 0 --num-workers 8 "$@" --countries "${D10[@]}"
}

for spec in "$S2_B3 s2_b3_augmax_full" "$S2_B7 s2_b7_augmax_full"; do
    set -- $spec; ckpt="$1"; stem="$2"
    run "$ckpt" "$stem" nows_notta
    run "$ckpt" "$stem" nows_tta   --tta
    run "$ckpt" "$stem" ws_notta   --watershed
    run "$ckpt" "$stem" ws_tta     --watershed --tta
done
echo "done."
