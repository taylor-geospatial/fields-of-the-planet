#!/bin/bash
#SBATCH --job-name=ftw-share-copy
#SBATCH --partition=cpu_amd
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/share_copy_%j.out
#SBATCH --error=logs/share_copy_%j.err

# Copy the final Planet FTW dataset (per-country patches + aligned eval dirs +
# index) into the shared project dir for agiri1. Excludes redundant release
# tarballs (bulk/) and manifests/caches (_global/). rsync is resumable.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SRC="data/planet/"
DST="/projects/bgtj/ftw-planet/data/planet/"
GRP="tgirails_bgtj"

mkdir -p "$DST"

rsync -a --info=progress2 \
  --exclude='bulk/' \
  --exclude='_global/' \
  "$SRC" "$DST"

# Make readable/writable by the project group; setgid on dirs so new files
# inherit the group; block other.
chgrp -R "$GRP" "$DST"
chmod -R g+rwX,o-rwx "$DST"
find "$DST" -type d -exec chmod g+s {} \;

echo "DONE: copied Planet dataset to $DST"
du -sh "$DST"
