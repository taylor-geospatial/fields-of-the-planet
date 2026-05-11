#!/bin/bash
#SBATCH --job-name=ftw-recover
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/recover_%j.out
#SBATCH --error=logs/recover_%j.err

# Recover by unzipping the existing data/<country>.zip files into
# data/ftw/<country>/ (skipping ones already populated), then re-indexing.
# Bypasses the `ftw data download` CLI entirely because that command deletes
# data/ftw/ before unpacking, which is destructive on retry.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/ftw

ALL=(austria belgium brazil cambodia corsica croatia denmark estonia finland france
     germany india kenya latvia lithuania luxembourg netherlands portugal rwanda
     slovakia slovenia south_africa spain sweden vietnam)

unzip_country() {
    local c="$1"
    local zip="data/$c.zip"
    local dest="data/ftw/$c"
    if [ ! -f "$zip" ]; then
        echo "[!] zip missing for $c — would need re-download" >&2
        return 1
    fi
    if [ -f "$dest/chips_$c.parquet" ]; then
        echo "[$(date)] $c already unpacked — skip"
        return 0
    fi
    echo "[$(date)] unzipping $c"
    mkdir -p "$dest"
    unzip -q -o "$zip" -d "$dest"
    return 0
}

export -f unzip_country
# Unzip in parallel — IO-bound, but cheap.
printf "%s\n" "${ALL[@]}" | xargs -n1 -P 8 -I{} bash -c 'unzip_country "$@"' _ {}

# Index every country directory.
for c in "${ALL[@]}"; do
    if [ -f "data/ftw/$c/chips_$c.parquet" ]; then
        if [ ! -f "data/ftw/$c/index.jsonl" ]; then
            echo "[$(date)] indexing $c"
            uv run --no-sync scripts/download_ftw.py --country "$c" --root data/ftw --skip-download
        else
            echo "[$(date)] index exists for $c"
        fi
    else
        echo "[!] missing parquet for $c — skipping index" >&2
    fi
done

echo "[$(date)] recover complete"
echo "indexed countries: $(ls data/ftw/*/index.jsonl 2>/dev/null | wc -l)"
