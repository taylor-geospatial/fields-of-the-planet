"""Upload the per-country tar shards to a HuggingFace dataset repo.

Mirrors ``data/planet/bulk/*.tar`` to ``<repo>/dataset/<country>.tar``. Each tar
is uploaded in its own commit, and shards already present on the repo at the
same byte size are skipped -- so a re-run (e.g. after an sbatch timeout) resumes
where it left off instead of re-pushing ~96 GiB.

The small README.md / index.parquet are uploaded separately. Auth via the
HF_TOKEN env var.

Example:
    HF_TOKEN=... uv run --with hf_xet scripts/pipeline/upload_hf.py
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="taylor-geospatial/ftw-planet")
    p.add_argument("--bulk-dir", type=Path, default=Path("data/planet/bulk"))
    p.add_argument("--path-in-repo", default="dataset", help="Repo subdir for the tars.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api = HfApi()

    tars = sorted(args.bulk_dir.glob("*.tar"))
    if not tars:
        raise SystemExit(f"no tars found under {args.bulk_dir}")

    # Remote sizes so we can skip shards that are already fully uploaded. List from
    # the repo root (always present) rather than the dataset/ subdir, which 404s
    # until the first tar lands.
    prefix = f"{args.path_in_repo}/"
    remote = {
        f.path: f.size
        for f in api.list_repo_tree(args.repo_id, repo_type="dataset", recursive=True)
        if f.path.startswith(prefix) and f.path.endswith(".tar")
    }

    print(f"{len(tars)} local tars -> {args.repo_id}/{args.path_in_repo}/")
    uploaded = skipped = 0
    for tar in tars:
        dst = f"{args.path_in_repo}/{tar.name}"
        local_size = tar.stat().st_size
        if remote.get(dst) == local_size:
            print(f"  [skip] {tar.name} (already {local_size} B on repo)")
            skipped += 1
            continue
        print(f"  [up]   {tar.name} ({local_size / 1024**3:.1f} GiB) ...")
        api.upload_file(
            path_or_fileobj=str(tar),
            path_in_repo=dst,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {tar.name}",
        )
        uploaded += 1

    print(f"done. uploaded {uploaded}, skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
