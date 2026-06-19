#!/bin/bash
#SBATCH --job-name=ftw-metric-example
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/metric_example_%j.out
#SBATCH --error=logs/metric_example_%j.err

# Renders the worked-example metrics figure (CPU inference of the Planet
# PRUE-FTP-B3 model on a single patch). Pass COUNTRY to override.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

COUNTRY="${COUNTRY:-croatia}"

uv run --no-sync python paper/scripts/make_metric_example.py --country "$COUNTRY"

# Light PNG preview for review (negligible cost).
uv run --no-sync --with pdf2image python -c "
from pdf2image import convert_from_path
convert_from_path('paper/figs/metric_example.pdf', dpi=200)[0].save('logs/metric_example.png')
print('preview -> logs/metric_example.png')
" || true
