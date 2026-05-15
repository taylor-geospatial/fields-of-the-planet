# FTW-Planet

Paired PlanetScope SR scenes (two windows per AOI — early- and peak-season)
co-registered with Fields of The World v2 field-boundary labels, across
25 countries.

- 66,584 patches, 25 countries
- 52,235 patches with both windows passing UDM2 usability (`usable_pair = True`)
- Imagery: PlanetScope `ortho_analytic_4b_sr`, 4 bands (B/G/R/NIR), 3 m GSD,
  native UTM, `uint16` (reflectance = DN / 10000)
- Labels: 3 classes — 0 background, 1 field interior, 2 field boundary;
  `uint8` with NBITS=2; boundaries rasterized with `all_touched=True`
  to match the FTW originals.

## Layout

```
s3://us-west-2.opendata.source.coop/ftw/ftw-planet/
├── README.md
├── index.parquet           # GeoParquet 1.1, one row per patch
└── dataset/
    ├── austria.tar
    ├── ...
    └── vietnam.tar         # 25 country shards, ~94 GiB total
```

Each tar is a WebDataset shard with four files per `patch_id`:

```
<pid>.window_a.tif    PlanetScope SR, window A
<pid>.window_b.tif    PlanetScope SR, window B
<pid>.label.tif       3-class label
<pid>.json            metadata (mirrors the index row)
```

Tars are uncompressed; the TIFFs inside are ZSTD-22. They stream as
WebDataset shards and also extract cleanly with `tar -xf <country>.tar`.

## Reading the index

```python
import geopandas as gpd

gdf = gpd.read_parquet("s3://us-west-2.opendata.source.coop/ftw/ftw-planet/index.parquet")
clean = gdf[gdf.usable_pair & (gdf.cloud_cover_a < 0.05) & (gdf.cloud_cover_b < 0.05)]
```

The index is GeoParquet 1.1 with a `bbox` covering struct and is
Hilbert-sorted into 14 row groups, so spatial queries from DuckDB /
duckdb-wasm can prune row groups by bbox without parsing WKB:

```sql
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;

SELECT patch_id, country
FROM 's3://us-west-2.opendata.source.coop/ftw/ftw-planet/index.parquet'
WHERE bbox.xmin > -10 AND bbox.xmax < 25
  AND bbox.ymin > 35  AND bbox.ymax < 60
  AND usable_pair;
```

## Index columns

Identity / geometry:

| column | type | notes |
|---|---|---|
| `patch_id` | str | unique within country |
| `country` | str | one of 25 slugs |
| `geometry` | polygon | EPSG:4326 patch footprint |
| `crs` | str | native UTM CRS of the tifs (e.g. `EPSG:32636`) |
| `bounds_4326` | float[4] | `[minx, miny, maxx, maxy]` convenience field |

Paths (relative to the tar / planet root):

| column | example |
|---|---|
| `image_a_path` | `rwanda/window_a/1592589.tif` |
| `image_b_path` | `rwanda/window_b/1592589.tif` |
| `label_path` | `rwanda/labels/1592589.tif` |

Scene provenance, per window suffix `_a` / `_b`:

| column | notes |
|---|---|
| `item_id_{a,b}` | PlanetScope item ID |
| `scene_date_{a,b}` | UTC acquisition timestamp |
| `cloud_cover_{a,b}` | scene-level fraction in [0,1] |
| `coverage_{a,b}` | AOI coverage of the source scene |
| `source_{a,b}` | source product / pipeline tag |

Per-patch UDM2 statistics (fraction of pixels in the patch), per window:

| column | meaning |
|---|---|
| `udm2_clear_{a,b}` | clear sky |
| `udm2_cloud_{a,b}` | cloud |
| `udm2_shadow_{a,b}` | cloud shadow |
| `udm2_light_haze_{a,b}` | light haze |
| `udm2_heavy_haze_{a,b}` | heavy haze |
| `udm2_snow_{a,b}` | snow / ice |
| `udm2_unusable_{a,b}` | UDM2 unusable mask |
| `udm2_confidence_mean_{a,b}` | mean UDM2 confidence band |
| `udm2_usable_flag_{a,b}` | bool — derived per-patch quality |

FTW season metadata:

| column | notes |
|---|---|
| `ftw_target_date_{a,b}` | target acquisition date for each window |
| `ftw_season_start` | growing-season start (per FTW) |
| `ftw_season_end` | growing-season end (per FTW) |

Quality:

| column | type | notes |
|---|---|---|
| `usable_pair` | bool | both windows pass UDM2 usability — the primary training subset |

## Licensing

Imagery is © Planet Labs PBC. The included AOIs were exported under the
NICFI / research program — refer to those terms for redistribution.
FTW v2 polygons are CC-BY-4.0; see `fieldsoftheworld/ftw-baselines` for
source terms.
