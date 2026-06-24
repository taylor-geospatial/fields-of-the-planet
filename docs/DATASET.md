# Fields of the Planet (FTP) dataset

Published at `s3://us-west-2.opendata.source.coop/ftw/ftw-planet/`.

- 66,584 patches (two seasonal windows each = 133,168 image-window pairs) across 24 countries / 25 labeled regions (Corsica shipped separately from mainland France). These are the patches successfully paired with PlanetScope, drawn from the 70,484 labeled FTW patches (140,968 patch-window targets); the 3,900 patches (7,800 targets) lacking a usable cloud-screened scene are dropped.
- 52,235 patches with `usable_pair = True` (both windows pass UDM2)
- Imagery: PlanetScope `ortho_analytic_4b_sr`, 4 bands (B/G/R/NIR), 3 m GSD, native UTM, `uint16` (reflectance = DN / 10000)
- Labels: 3-class — 0 background, 1 field interior, 2 field boundary; `uint8` NBITS=2; boundaries rasterized `all_touched=True`

## Layout

```
s3://us-west-2.opendata.source.coop/ftw/ftw-planet/
├── README.md
├── index.parquet           # GeoParquet 1.1, one row per patch
└── dataset/
    ├── austria.tar
    ├── ...
    └── vietnam.tar         # 25 region shards (24 countries; Corsica separate), ~94 GiB total
```

Each tar is a WebDataset shard, five files per `patch_id`:

```
<pid>.window_a.tif        PlanetScope SR, window A
<pid>.window_b.tif        PlanetScope SR, window B
<pid>.label.tif           3-class label
<pid>.polygons.parquet    true FTW field polygons, clipped to the patch
<pid>.json                metadata (mirrors index row)
```

`<pid>.polygons.parquet` is GeoParquet of the original FTW vector field boundaries, reprojected to the patch's UTM grid and clipped to its bounds — the same vector source the `.label.tif` raster is burned from. Columns: `id`, `geometry`, `area_ha` (true planimetric area), plus any of `crop_id`/`crop_name`/`area`/`perimeter` present in the source. Patches with no fields carry an empty (0-row) GeoParquet, so every sample has the file.

Tars uncompressed; inner TIFFs ZSTD-22. Stream as WebDataset shards or extract with `tar -xf <country>.tar`.

To run training/eval locally, extract the tars into `data/planet/<country>/` so each patch resolves as `data/planet/<country>/window_{a,b}/<pid>.tif` with labels under `data/planet/<country>/labels/`.

## Reading the index

```python
import geopandas as gpd

gdf = gpd.read_parquet("s3://us-west-2.opendata.source.coop/ftw/ftw-planet/index.parquet")
clean = gdf[gdf.usable_pair & (gdf.cloud_cover_a < 0.05) & (gdf.cloud_cover_b < 0.05)]
```

Index is GeoParquet 1.1 with a `bbox` covering struct, Hilbert-sorted into 14 row groups. DuckDB prunes row groups by bbox without parsing WKB:

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

| column        | type     | notes                                                   |
| ------------- | -------- | ------------------------------------------------------- |
| `patch_id`    | str      | unique within country                                   |
| `country`     | str      | one of 25 region slugs (24 countries; Corsica separate) |
| `geometry`    | polygon  | EPSG:4326 patch footprint                               |
| `crs`         | str      | native UTM CRS of the tifs (e.g. `EPSG:32636`)          |
| `bounds_4326` | float[4] | `[minx, miny, maxx, maxy]` convenience field            |

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
