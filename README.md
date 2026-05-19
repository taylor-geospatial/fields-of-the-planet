# ftw-planet

Higher-resolution field-boundary segmentation. Pairs **Fields of the World v2** (FTW) Sentinel-2 patches with **PlanetScope** ~3 m SR imagery so models can predict field boundaries at ~3 m instead of 10 m.

Two seasonal windows per AOI (early- and peak-season) co-registered with FTW v2 labels across 25 countries.

## Setup

```bash
make install                 # uv sync --all-extras
cp .env.example .env         # then set PL_API_KEY=<your Planet Data API key>
```

Stack: Python 3.13, [uv](https://docs.astral.sh/uv/), ruff, ty, pytest, pre-commit. Makefile targets: `install`, `check`, `test`, `clean`.

## Repo layout

```
src/ftw_planet/      # package: datamodules, datasets, trainers, losses, pipeline helpers
scripts/             # dataset pipeline + training/eval entrypoints (see scripts/README.md)
  slurm/             # SLURM wrappers, one per pipeline phase
configs/prue/        # training configs (efnet3/5/7, crop sizes, loss variants)
data/                # FTW patches, Planet outputs, indices (gitignored)
logs/                # checkpoints + W&B runs (gitignored)
docs/                # profiling.md, planet-api-issues.md
paper/               # LaTeX source + figures
tests/
```

## Dataset

Published at `s3://us-west-2.opendata.source.coop/ftw/ftw-planet/`.

- 66,584 patches, 25 countries
- 52,235 patches with `usable_pair = True` (both windows pass UDM2)
- Imagery: PlanetScope `ortho_analytic_4b_sr`, 4 bands (B/G/R/NIR), 3 m GSD, native UTM, `uint16` (reflectance = DN / 10000)
- Labels: 3-class — 0 background, 1 field interior, 2 field boundary; `uint8` NBITS=2; boundaries rasterized `all_touched=True`

### Layout

```
s3://us-west-2.opendata.source.coop/ftw/ftw-planet/
├── README.md
├── index.parquet           # GeoParquet 1.1, one row per patch
└── dataset/
    ├── austria.tar
    ├── ...
    └── vietnam.tar         # 25 country shards, ~94 GiB total
```

Each tar is a WebDataset shard, four files per `patch_id`:

```
<pid>.window_a.tif    PlanetScope SR, window A
<pid>.window_b.tif    PlanetScope SR, window B
<pid>.label.tif       3-class label
<pid>.json            metadata (mirrors index row)
```

Tars uncompressed; inner TIFFs ZSTD-22. Stream as WebDataset shards or extract with `tar -xf <country>.tar`.

### Reading

```python
import geopandas as gpd

gdf = gpd.read_parquet("s3://us-west-2.opendata.source.coop/ftw/ftw-planet/index.parquet")
clean = gdf[gdf.usable_pair & (gdf.cloud_cover_a < 0.05) & (gdf.cloud_cover_b < 0.05)]
```

Index is GeoParquet 1.1 with `bbox` covering struct, Hilbert-sorted into 14 row groups. DuckDB / duckdb-wasm prune row groups by bbox without parsing WKB:

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;

SELECT patch_id, country
FROM 's3://us-west-2.opendata.source.coop/ftw/ftw-planet/index.parquet'
WHERE bbox.xmin > -10 AND bbox.xmax < 25
  AND bbox.ymin > 35  AND bbox.ymax < 60
  AND usable_pair;
```

### Index columns

Identity / geometry:

| column        | type     | notes                                          |
| ------------- | -------- | ---------------------------------------------- |
| `patch_id`    | str      | unique within country                          |
| `country`     | str      | one of 25 slugs                                |
| `geometry`    | polygon  | EPSG:4326 patch footprint                      |
| `crs`         | str      | native UTM CRS of the tifs (e.g. `EPSG:32636`) |
| `bounds_4326` | float[4] | `[minx, miny, maxx, maxy]` convenience field   |

Paths (relative to tar / planet root):

| column         | example                       |
| -------------- | ----------------------------- |
| `image_a_path` | `rwanda/window_a/1592589.tif` |
| `image_b_path` | `rwanda/window_b/1592589.tif` |
| `label_path`   | `rwanda/labels/1592589.tif`   |

Scene provenance, per window suffix `_a` / `_b`:

| column              | notes                            |
| ------------------- | -------------------------------- |
| `item_id_{a,b}`     | PlanetScope item ID              |
| `scene_date_{a,b}`  | UTC acquisition timestamp        |
| `cloud_cover_{a,b}` | scene-level fraction in [0,1]    |
| `coverage_{a,b}`    | AOI coverage of the source scene |
| `source_{a,b}`      | source product / pipeline tag    |

Per-patch UDM2 stats (pixel fraction), per window:

| column                       | meaning                          |
| ---------------------------- | -------------------------------- |
| `udm2_clear_{a,b}`           | clear sky                        |
| `udm2_cloud_{a,b}`           | cloud                            |
| `udm2_shadow_{a,b}`          | cloud shadow                     |
| `udm2_light_haze_{a,b}`      | light haze                       |
| `udm2_heavy_haze_{a,b}`      | heavy haze                       |
| `udm2_snow_{a,b}`            | snow / ice                       |
| `udm2_unusable_{a,b}`        | UDM2 unusable mask               |
| `udm2_confidence_mean_{a,b}` | mean UDM2 confidence band        |
| `udm2_usable_flag_{a,b}`     | bool — derived per-patch quality |

FTW season metadata:

| column                  | notes                                   |
| ----------------------- | --------------------------------------- |
| `ftw_target_date_{a,b}` | target acquisition date for each window |
| `ftw_season_start`      | growing-season start (per FTW)          |
| `ftw_season_end`        | growing-season end (per FTW)            |

Quality:

| column        | type | notes                                                          |
| ------------- | ---- | -------------------------------------------------------------- |
| `usable_pair` | bool | both windows pass UDM2 usability — the primary training subset |

## Pipeline

Full search → activate → extract → rasterize pipeline lives in `scripts/` with SLURM wrappers in `scripts/slurm/`. See [`scripts/README.md`](scripts/README.md) for phase-by-phase detail, artifact layout, and SLURM invocation.

## Training

Training is driven by `ftw model fit` (from `ftw-tools`) with a LightningCLI config from `configs/prue/`:

```bash
uv run ftw model fit -c configs/prue/ftw_planet_efnet3_crop512_v3_augmax.yaml
```

On SLURM, submit `scripts/slurm/train_prue.sbatch` (set `CONFIG=...` to pick a config; `CKPT_PATH=...` to resume). Configs in `configs/prue/` cover backbones (efnet3/5/7), crop sizes, and loss variants (cldice, boundary, sdf, framefield, augmax).

## Evaluation

```bash
uv run scripts/eval_planet.py --ckpt <ckpt> --out logs/eval_planet.csv
uv run scripts/polygon_metrics_eval.py ...
uv run scripts/viz_predictions.py ...
```

`eval_planet.py` reports per-country pixel IoU/precision/recall and object-level precision/recall/F1. For field-boundary work, lead with object F1 / polygon metrics; pixel IoU is secondary.

## Paper

LaTeX source under `paper/`. Build with `cd paper && make build` (requires TeX Live + latexmk).

## Citation

```bibtex
@misc{ftw-planet,
  author = {Corley, Isaac},
  title  = {FTW-Planet: PlanetScope companion to Fields of the World v2},
  year   = {2026},
  url    = {https://github.com/isaaccorley/ftw-planet}
}
```

## License / Data terms

Code: see `LICENSE`. Imagery is © Planet Labs PBC; included AOIs were exported under the NICFI / research program — refer to those terms for redistribution. FTW v2 polygons are CC-BY-4.0; see `fieldsoftheworld/ftw-baselines` for source terms.
