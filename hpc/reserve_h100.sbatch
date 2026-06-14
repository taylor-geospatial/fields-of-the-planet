#!/bin/bash
#SBATCH --job-name=ftw-reserve
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/reserve_%j.out
#SBATCH --error=logs/reserve_%j.err

mkdir -p logs

# Resubmit self 5 minutes before the 7-day limit so the chain never breaks.
_resubmit() {
    echo "$(date): resubmitting reserve job before timeout"
    sbatch --dependency=afterany:$SLURM_JOB_ID scripts/slurm/reserve_h100.sbatch
}
trap _resubmit USR1

echo "$(date): holding $SLURMD_NODENAME (job $SLURM_JOB_ID)"
sleep infinity &
wait
