"""Upload the per-country tar shards to a HuggingFace dataset repo.

Mirrors ``data/planet/bulk/*.tar`` to ``<repo>/dataset/<country>.tar``. Each tar
is uploaded in its own commit, and shards already present on the repo at the
same byte size are skipped -- so a re-run (e.g. after an sbatch timeout) resumes
where it left off instead of re-pushing ~96 GiB.

For multi-node parallelism, pass ``--slice-id i --slice-count n`` (or the
SLICE_ID / SLICE_COUNT env vars) so each node uploads a disjoint, largest-first
round-robin subset from its own NIC. Concurrent nodes committing to the same
branch occasionally race (HTTP 412); the LFS blob upload is already done by then,
so we just retry the cheap commit step.

The small README.md / index.parquet are uploaded separately. Auth via the
HF_TOKEN env var.

Example:
    HF_TOKEN=... uv run --with hf_xet scripts/pipeline/upload_hf.py            # all shards
    SLICE_ID=0 SLICE_COUNT=5 uv run --with hf_xet scripts/pipeline/upload_hf.py  # one slice
"""

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, RepoFile
from huggingface_hub.errors import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="taylor-geospatial/ftw-planet")
    p.add_argument("--bulk-dir", type=Path, default=Path("data/planet/bulk"))
    p.add_argument("--path-in-repo", default="dataset", help="Repo subdir for the tars.")
    p.add_argument("--slice-id", type=int, default=int(os.environ.get("SLICE_ID", "0")))
    p.add_argument("--slice-count", type=int, default=int(os.environ.get("SLICE_COUNT", "1")))
    return p.parse_args()


def _upload_with_retry(api: HfApi, tar: Path, dst: str, repo_id: str, attempts: int = 8) -> None:
    """Upload one shard, retrying only the commit on cross-node 412 conflicts."""
    for attempt in range(1, attempts + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(tar),
                path_in_repo=dst,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {tar.name}",
            )
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status not in (409, 412) or attempt == attempts:
                raise
            backoff = min(2**attempt, 30)
            print(
                f"    commit conflict ({status}) on {tar.name}, retry {attempt}/{attempts} in {backoff}s"
            )
            time.sleep(backoff)
        else:
            return


def main() -> int:
    args = parse_args()
    api = HfApi()

    tars = sorted(args.bulk_dir.glob("*.tar"))
    if not tars:
        raise SystemExit(f"no tars found under {args.bulk_dir}")

    # Largest-first round-robin so the multi-GB shards spread across slices.
    tars.sort(key=lambda t: t.stat().st_size, reverse=True)
    mine = [t for i, t in enumerate(tars) if i % args.slice_count == args.slice_id]

    # Remote sizes so we can skip shards that are already fully uploaded. List from
    # the repo root (always present) rather than the dataset/ subdir, which 404s
    # until the first tar lands.
    prefix = f"{args.path_in_repo}/"
    remote = {
        f.path: f.size
        for f in api.list_repo_tree(args.repo_id, repo_type="dataset", recursive=True)
        if isinstance(f, RepoFile) and f.path.startswith(prefix) and f.path.endswith(".tar")
    }

    print(
        f"slice {args.slice_id}/{args.slice_count}: {len(mine)} of {len(tars)} tars "
        f"-> {args.repo_id}/{prefix}"
    )
    uploaded = skipped = 0
    for tar in mine:
        dst = f"{prefix}{tar.name}"
        local_size = tar.stat().st_size
        if remote.get(dst) == local_size:
            print(f"  [skip] {tar.name} (already {local_size} B on repo)")
            skipped += 1
            continue
        print(f"  [up]   {tar.name} ({local_size / 1024**3:.1f} GiB) ...")
        _upload_with_retry(api, tar, dst, args.repo_id)
        uploaded += 1

    print(f"slice {args.slice_id}: done. uploaded {uploaded}, skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
