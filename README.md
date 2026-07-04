# Fields of the Planet (FTP)

Higher-resolution field-boundary segmentation. **Fields of the Planet (FTP)**
pairs every **Fields of the World v2** (FTW) Sentinel-2 patch with a
co-registered **PlanetScope** ~3 m surface-reflectance image, so models predict
field boundaries at 3 m instead of 10 m. Two seasonal windows per patch
(planting and harvest) across 24 countries / 25 labeled regions.

## Install

```bash
make install        # uv sync --all-extras
```

This installs the `ftw-planet` command. Run all commands from the repository
root. Stack: Python 3.13, [uv](https://docs.astral.sh/uv/), PyTorch Lightning.

## Quickstart

With the dataset extracted under `data/` (see [Dataset](#dataset)):

```bash
ftw-planet check-data                                              # validate the data layout
ftw-planet reproduce --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt
ftw-planet train configs/prue/ftw_planet_efnet3_crop512_v3_augmax_full.yaml
```

`reproduce` evaluates the released EfficientNet-B3 checkpoint over the ten
dense-label held-out countries with watershed + D4 test-time augmentation, then
prints the polygon-level metrics next to the published paper numbers. Full
walkthrough: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Models

`configs/prue/` holds the four reported models, each a U-Net with an
EfficientNet encoder trained via `ftw model fit`:

| Config                                          | Sensor          | Backbone                     |
| ----------------------------------------------- | --------------- | ---------------------------- |
| `ftw_planet_efnet3_crop512_v3_augmax_full.yaml` | PlanetScope 3 m | EfficientNet-B3 (main model) |
| `ftw_planet_efnet7_crop512_v3_augmax_full.yaml` | PlanetScope 3 m | EfficientNet-B7              |
| `ftw_s2_efnet3_crop256_v3_augmax_full.yaml`     | Sentinel-2 10 m | EfficientNet-B3 (baseline)   |
| `ftw_s2_efnet7_crop256_v3_augmax_full.yaml`     | Sentinel-2 10 m | EfficientNet-B7 (baseline)   |

## Evaluation

Models are scored as parcel-recovery systems on **vectorized** predictions, not
pixel maps. `ftw-planet eval` runs `scripts/eval/postprocess_eval.py`
(watershed + TTA) and `scripts/eval/polygon_metrics_eval.py`, reporting panoptic
quality (PQ/SQ/RQ), object F1 over the COCO IoU grid, and meter-scale
matched-boundary error. Pixel IoU is reported only for continuity.

## Dataset

FTP is published on Hugging Face (`taylor-geospatial/ftw-planet`) and Source
Cooperative (`s3://us-west-2.opendata.source.coop/ftw/ftw-planet/`):

- **66,584 patches** (two seasonal windows each = 133,168 image-window pairs)
    across 24 countries / 25 labeled regions, drawn from the 70,484 labeled FTW
    patches.
- **Imagery:** PlanetScope `ortho_analytic_4b_sr`, 4 bands (B/G/R/NIR), ~3 m
    GSD, native UTM, `uint16` (reflectance = DN / 10000).
- **Labels:** 3-class raster — 0 background, 1 field interior, 2 field boundary
    (`uint8`). The original FTW vector polygons ship alongside as GeoParquet.

The release is one WebDataset tar per region (25 shards, ~94 GiB) plus a
GeoParquet `index.parquet` (one row per patch). Each patch has five members:

```
<pid>.window_a.tif        PlanetScope SR, planting window
<pid>.window_b.tif        PlanetScope SR, harvest window
<pid>.label.tif           3-class label
<pid>.polygons.parquet    original FTW field polygons, clipped to the patch
<pid>.json                per-patch metadata
```

For local training/eval, extract each tar into `data/planet/<country>/` so a
patch resolves as `data/planet/<country>/window_{a,b}/<pid>.tif` with labels
under `data/planet/<country>/labels/`. `ftw-planet check-data` validates the
layout.

## Paper

LaTeX source under `paper/`. Build with `cd paper && make build` (requires
TeX Live + latexmk).

## Citation

```bibtex
@misc{ftw-planet,
  author = {Corley, Isaac},
  title  = {Fields of the Planet: A PlanetScope companion to Fields of the World v2},
  year   = {2026},
  url    = {https://github.com/taylor-geospatial/fields-of-the-planet}
}
```

## License / data terms

Code: see `LICENSE`. Imagery is © Planet Labs PBC (non-commercial terms); FTW v2
polygons are CC-BY-4.0 — see `fieldsoftheworld/ftw-baselines` for source terms.
