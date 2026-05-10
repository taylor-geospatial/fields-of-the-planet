# ftw-planet

Higher-resolution field boundary segmentation by pairing the **Fields of the World** (FTW) benchmark with **PlanetScope** imagery.

FTW labels are 10 m Sentinel-2 patches with two timestamps (`window_a` planting, `window_b` harvest) and a field-boundary mask. This repo finds the nearest cloud-free PlanetScope (~3 m) scene for each patch's window, crops it to the patch bounds, and saves it as a drop-in higher-resolution replacement for the S2 imagery — so we can train segmentation models that produce ~3 m field boundaries.

## Setup

```bash
make install                 # uv sync --all-extras
cp .env.example .env         # then add your PL_API_KEY
```

## Pipeline

```bash
# 1. Download a country's FTW patches + build a per-patch index (bounds, dates).
uv run scripts/download_ftw.py --country rwanda --root data/ftw

# 2. For each patch, search Planet, pick nearest cloud-free PSScene, crop to patch bounds.
uv run scripts/match_planet.py \
    --ftw-root data/ftw \
    --country rwanda \
    --out data/planet \
    --search-days 14 \
    --max-cloud-cover 0.1
```

Outputs land in `data/planet/<country>/<patch_id>_{a,b}.tif`, georeferenced and aligned to the FTW patch grid (resampled to PlanetScope's native ~3 m).

## Stack

- Python ≥ 3.13, [uv](https://docs.astral.sh/uv/) for dep management
- ruff + ty + pre-commit for lint/format/types
- pytest for tests
