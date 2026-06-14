# Fields of the Planet (FTP)

Higher-resolution field-boundary segmentation. **Fields of the Planet (FTP)**
pairs every **Fields of the World v2** (FTW) Sentinel-2 patch with a
co-registered **PlanetScope** ~3 m surface-reflectance image, so models can
predict field boundaries at 3 m instead of 10 m. Two seasonal windows per AOI
(early- and peak-season) across 25 countries.

## Install

```bash
make install        # uv sync --all-extras
```

Stack: Python 3.13, [uv](https://docs.astral.sh/uv/), ruff, ty, pytest,
pre-commit. Makefile targets: `install`, `check`, `test`, `clean`.

## Quickstart

Run from the repository root, with the dataset under `data/` (see
[docs/DATASET.md](docs/DATASET.md)):

```bash
ftw-planet check-data                                              # validate the data layout
ftw-planet reproduce --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt
ftw-planet train configs/prue/ftw_planet_efnet3_crop512_v3_augmax_full.yaml
```

`reproduce` evaluates the released EfficientNet-B3 checkpoint over the
dense-label held-out countries and prints the headline metrics next to the
paper numbers. Full walkthrough: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repo layout

```
src/ftw_planet/      # package: datasets, datamodules, trainers, losses, cli
configs/prue/        # training configs (efnet3/5/7, crop sizes, loss variants)
scripts/eval/        # evaluation entrypoints
scripts/pipeline/    # dataset-building pipeline (Planet API; internal)
hpc/                 # optional SLURM wrappers (RAILS); the CLI is the supported path
docs/                # REPRODUCE.md, DATASET.md, profiling.md, planet-api-issues.md
paper/               # LaTeX source + figures
tests/
```

## Dataset

The FTP dataset (66,584 patches, 25 countries) is described in
[docs/DATASET.md](docs/DATASET.md): S3 layout, the GeoParquet index schema,
and how to read it. To build it from scratch from the Planet API, see
[scripts/README.md](scripts/README.md).

## Paper

LaTeX source under `paper/`. Build with `cd paper && make build` (requires
TeX Live + latexmk).

## Citation

```bibtex
@misc{ftw-planet,
  author = {Corley, Isaac},
  title  = {Fields of the Planet: A PlanetScope companion to Fields of the World v2},
  year   = {2026},
  url    = {https://github.com/isaaccorley/ftw-planet}
}
```

## License / data terms

Code: see `LICENSE`. Imagery is © Planet Labs PBC; included AOIs were exported
under the NICFI / research program — refer to those terms for redistribution.
FTW v2 polygons are CC-BY-4.0; see `fieldsoftheworld/ftw-baselines` for source
terms.
