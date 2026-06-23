# FTP results bundle

Standalone export of the headline field-boundary metrics reported in the
*Fields of the Planet* (FTP) paper, on the **true-GT / native-GSD** protocol.
Hand this folder to anyone who asks for "the results."

Regenerate with:

```
uv run python paper/scripts/results_bundle.py
```

The two CSVs are script-built from the same canonical logs as the paper's
`tab:polygon_metrics` (via `paper/scripts/polygon_metrics_table.py`), so they
match Table 1 exactly and cannot drift. Full per-float provenance lives in
`paper/scripts/PROVENANCE.md`.

## Scoring protocol

Segmentation rows are scored against the **true FTW vector polygons** (not a
rasterized label mask) at **each sensor's native ground resolution**: PlanetScope
at 3 m, Sentinel-2 capped to its 10 m grid before matching (`--score-gsd-m 10`),
so an upsampled output grid cannot manufacture sub-pixel boundaries the sensor
never resolved. Metrics: panoptic quality `PQ = SQ x RQ`, object F1 (`= RQ` at
IoU 0.5), threshold-averaged F1 over IoU {.5:.95}, boundary chamfer in meters,
and `|dN|/N` polygon-count error. Definitions: paper Appendix A.

**Metrics are fractions in `[0,1]`** here (e.g. PQ `0.3550`); the paper shows them
`x100` (`35.5`). Boundary error is in meters.

## Files

- **`macro_summary.csv`** — one row per method, macro-averaged over the 10 dense
    held-out countries (= paper Table 1). Columns: `method, backbone, pq, sq, rq, f1_5_95, dN_over_N, bnd_mean_m, bnd_p95_m, pixel_iou, pq_small, pq_medium, pq_large`. The two DelineateAnything rows are zero-shot baselines retained on
    their rasterized-GT scores (no size bins, hence blank `pq_{small,medium,large}`);
    pixel IoU is rasterized at each sensor's native grid and is **not** comparable
    across the 3 m / 10 m resolutions.
- **`per_country_metrics.csv`** — per-country true-GT overall metrics plus
    PQ-by-size, for the four segmentation models (40 rows = 4 models x 10 countries).
    Columns add `condition` (the `logs/resolution_ablation/<dir>` it came from),
    `country`, `n_patches`, `n_pred_mean`, `n_gt_mean`.

## The 10 dense held-out countries

`belgium, cambodia, croatia, germany, latvia, lithuania, portugal, slovenia, south_africa, sweden`. Kenya is presence-only (untrusted background) and is
excluded from every supervised macro.

## Headline (paper scale, x100)

| Model                  | Bb.    | PQ       | RQ (obj F1) | Bd. err (m) | PQ small | PQ large |
| ---------------------- | ------ | -------- | ----------- | ----------- | -------- | -------- |
| FTW-PRUE+ (S2)         | B3     | 21.0     | 28.9        | 18.6        | 5.8      | 33.8     |
| FTW-PRUE+ (S2)         | B7     | 24.2     | 32.8        | 14.4        | 7.5      | 37.7     |
| **FTP-PRUE+ (Planet)** | **B3** | **35.5** | **46.2**    | **7.4**     | **15.7** | **52.0** |
| FTP-PRUE+ (Planet)     | B7     | 35.4     | 46.1        | 7.4         | 15.6     | 50.9     |

## Source map

| Output                                                  | Source                                                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `macro_summary.csv` PQ/SQ/RQ/F1 + PQ-by-size (seg rows) | `logs/resolution_ablation/{s2b3_10m,s2nat10,planet3m,planetb7_3m}/` (true-GT, native-GSD), macro over the dense-10 |
| `macro_summary.csv` boundary chamfer                    | `logs/polygon_metrics/s2_*_native256.csv` (S2), `logs/repro_eval/polygon_metrics.csv` (Planet)                     |
| `macro_summary.csv` pixel IoU                           | `logs/postproc_ablation/*_ws_tta.csv`, `logs/repro_eval/pp_ws_tta.csv`                                             |
| `macro_summary.csv` DelineateAnything rows              | `logs/polygon_metrics/delineate_{x,s}_planet.csv` (rasterized GT, not re-scored)                                   |
| `per_country_metrics.csv`                               | per-country `logs/resolution_ablation/<condition>/<country>.csv` (+ `.bins.csv`)                                   |
