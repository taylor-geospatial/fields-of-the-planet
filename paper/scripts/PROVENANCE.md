# Table & figure provenance

Maps every numeric float in `paper/main.tex` to the script that builds it and
the canonical source it reads. Script-generated tables rebuild with one command:

```
uv run --no-sync python paper/scripts/build_tables.py
```

All evaluation CSVs are produced by `scripts/eval/polygon_metrics_eval.py`
(polygon metrics + `--area-bins`), `per_patch_metrics.py`, and
`postprocess_eval.py`. Sentinel-2 object metrics use the canonical upsampled
eval (`--upsample-to 512`, `resize_factor=2`); boundary chamfer is reported at
each model's native grid.

## Script-generated tables (`\input{figs/*.tex}`)

| Float                 | Generator                      | Source CSV(s)                                                                                                                         |
| --------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `tab:polygon_metrics` | `polygon_metrics_table.py`     | Polygon metrics + \`                                                                                                                  |
| `tab:area_bins`       | `area_bins_table.py`           | `logs/area_bins/{planet_b3,s2_b7,s2_b3}.csv.bins.csv`                                                                                 |
| `tab:heldout`         | `heldout_results_table.py`     | `logs/postproc_ablation/s2_{b3,b7}_augmax_full_upsampled_{nows_notta,nows_tta,ws_notta,ws_tta}.csv`; `logs/repro_eval/pp_*.csv` (FTP) |
| `tab:heldout_pc`      | `heldout_per_country_table.py` | `logs/postproc_ablation/s2_b3_augmax_full_upsampled_ws_tta.csv`; `logs/repro_eval/pp_ws_tta.csv`                                      |
| `tab:full_data`       | `full_data_table.py`           | `logs/repro_eval/pp_ws_tta.csv` (FTP); released PRUE values from \\cite{muhawenayo2026prue}                                           |

### Planet B3 eval-run note

The dense-10 headline (`tab:polygon_metrics`, `tab:area_bins`, etc.) reads the
**most recent** Planet B3 eval, `logs/repro_eval/` (2026-06-13), giving PQ
`36.0`. The 23-region per-region breakdown (`fig:per_country_bars`,
per-region head-to-head prose) reads
`logs/polygon_metrics/planet_b3_augmax_full_22.csv` (2026-05-21), the only run
with all 23 regions; its dense-10 subset is PQ `35.6`. The two runs use the same
`planet_efnet3_augmax_full_best.ckpt` and differ by `<=0.4` PQ on shared
countries (eval-time settings). `35.6` is never quoted in the paper, so no
reader-visible number depends on the difference; the 10-country (`36.0`) and
23-region (`37.9`) macros are different country sets by construction.

## Script-generated figures

| Float                       | Generator                                                                                 | Source                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fig:aug_ablation`          | `aug_ablation.py`                                                                         | `paper/scripts/output/aug_ablation_heldout10.csv` (Kenya-excluded, matching the paper's supervised-macro protocol; from `logs/heldout/*.csv`, `logs/repro_eval/pp_ws_tta.csv`). Bars 1--4 are CC-BY-subset models on the 10 held-out (OOD) countries; the final bar is the in-distribution full-data model.                                                                                                                               |
| `fig:per_country_bars`      | `per_country_pq_objf1.py`                                                                 | split panels: left uses per-region $\Delta$PQ from `logs/polygon_metrics/planet_b3_augmax_full_22.csv` (Planet, all-23-region run; the 37.9 macro) vs `logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv` (S2-B7); right uses FTW-official per-region Obj-F1 (`logs/fulldata_eval/`, `logs/ftw_official/b7_*.csv`; Brazil n/a). If logs are absent, the script falls back to embedded plotted deltas matching the committed figure. |
| `fig:improvement`           | `improvement_figure.py`                                                                   | `logs/per_patch/{planet_b3,s2_b7}.csv`                                                                                                                                                                                                                                                                                                                                                                                                    |
| `fig:metric_example`        | `metric_example.py`                                                                       | per-patch vectorized predictions (FTP-PRUE+)                                                                                                                                                                                                                                                                                                                                                                                              |
| `fig:qualitative`, `*_appx` | `qualitative_main.py`, `qualitative_raw_appendix.py`, `qualitative_instances_appendix.py` | held-out patch predictions (both sensors). `qualitative_raw_appendix` rows are seven dense per-held-out-country patches picked from `logs/per_patch/{planet_b3,s2_b7}.csv` where Planet's per-patch object F1 beats S2 (+25 to +51 pp); disjoint from the main figure (`qualitative_main`) and `qualitative_instances_appendix`.                                                                                                          |

## Hand-entered tables (no generator)

These are typed directly in `main.tex`; a comment above each points to its
source. Verify against the source before camera-ready.

| Float                                       | Source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tab:scope`                                 | dataset build manifest (`data/_global/` index: per-region patch/window counts, success rates)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `tab:udm2`                                  | per-class coverage-threshold fractions, re-derived from the per-patch stats in `data/planet/index.parquet` (the shipped geoparquet; the raw UDM2 tifs are not retained). Pool both windows `v = pd.concat([df.udm2_<band>_a, df.udm2_<band>_b]).dropna()` (n=129,490 tiles), then report `100*(v>t).mean()` for t in {0, 0.5, 0.9}. (Earlier drafts reported median/p90/p99/max; the percentiles read as alarming for the zero-inflated haze/cloud classes, so the table now reports the fraction of tiles exceeding each coverage level.) Per-region usable rate = per-window mean of `udm2_usable_flag_{a,b}` (flag = clear$\ge$95% and unusable$\le$5%). |
| `tab:upsampled_s2_main`, `tab:upsampled_s2` | the upsampled-S2 control eval: native-256 (`s2_b3_augmax_full_native256.csv`), upsample-512-at-eval (`s2_b3_augmax_full_upsampled_22.csv`), and trained-upsampled-512 (`s2_upsampled_b3_augmax_full.csv`) dense-10 macros; Planet row from `logs/repro_eval/`. PQ/SQ/RQ/F1 = `pq`/`pq_sq`/`pq_rq`/`ap_5_95`; boundary is nan-mean over countries (Portugal native-256 chamfer is NaN, so n=9). The PQ/SQ/RQ/F1 of the "upsample-at-eval" row equal the `tab:polygon_metrics` FTW-PRUE+ B3 row; the two tables mirror each other.                                                                                                                            |
| `tab:ablation_summary`                      | effect sizes from the recipe ablations; per-row mapping (config + eval CSV + metric + baseline) in **tab:ablation_summary provenance** below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### tab:ablation_summary provenance

Metric throughout is **held-out-10 (dense, Kenya-excluded) `object_ws_f1` $\times100$**
(same country set and column as `fig:aug_ablation`). Deltas are vs the baseline
named in each row. "config-only" rows have a committed `configs/prue/*.yaml` but
no committed eval CSV — rerun the config to reproduce the magnitude.

| Row                       | $\Delta$ | Config                                        | Eval CSV(s)                                                               | Baseline                                                              | Reproduces?         |
| ------------------------- | -------- | --------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------- |
| PRUE preprocess+resize    | $+2.8$   | (PRUE recipe)                                 | `logs/heldout/b3base_aug_best.csv` (31.9)                                 | `b3base_best.csv` (29.1)                                              | yes                 |
| `swap_order` + $\gamma$   | $+1.0$   | —                                             | `logs/heldout/v3_augplus.csv` (32.9)                                      | `b3base_aug_best.csv` (31.9)                                          | yes                 |
| Geometry/noise (PRUE+)    | $+4.5$   | —                                             | `logs/heldout/v3_augmax_ws_tta.csv` (37.4)                                | `v3_augplus.csv` (32.9)                                               | yes                 |
| B3 $\to$ B7               | $\sim0$  | `..._efnet7_crop512_v3_augmax*.yaml`          | `logs/fulldata_eval/s2_b7_*`, `logs/heldout/v3_augmax_b7_full_ws_tta.csv` | B3 PRUE+                                                              | yes                 |
| Soft clDice               | failed   | `..._efnet3_crop512_cldice.yaml`              | config-only (run diverged)                                                | —                                                                     | config-only         |
| SDF aux head (with PRUE+) | $-3.1$   | `..._efnet3_crop512_v3_augmax_full_sdf.yaml`  | `logs/heldout/v3_augmax_b3_sdf_full_ws_tta.csv` (42.03)                   | `logs/repro_eval/pp_ws_tta.csv` (45.16, B3 full PRUE+ ws+tta, no SDF) | yes ($-3.13$)       |
| Frame-field head          | $-1.7$   | `..._efnet3_crop512_framefield.yaml`          | `logs/heldout/b3ff_best.csv` (27.4)                                       | `b3base_best.csv` (29.1)                                              | yes ($-1.7$)        |
| CutMix (2px ignore)       | $-0.1$   | `..._efnet3_crop512_v2_cutmix.yaml`           | `logs/heldout/v2_cutmix.csv` (32.73)                                      | `v2_augplus.csv` (32.86)                                              | yes ($-0.13$)       |
| Curriculum dilation       | $-0.4$   | `..._efnet3_crop512_curriculum.yaml`          | config-only                                                               | PRUE+                                                                 | config-only         |
| Watershed                 | $+0.5$   | —                                             | `tab:heldout` (WS vs no-WS)                                               | no-WS                                                                 | yes (`tab:heldout`) |
| D4 TTA                    | $+0.9$   | —                                             | `tab:heldout` (TTA vs no-TTA)                                             | no-TTA                                                                | yes (`tab:heldout`) |
| Replicate padding         | $-8/-10$ | `..._efnet3_crop512_v3_augmax_replicate.yaml` | config-only                                                               | ignore-pad PRUE+                                                      | config-only         |
| Austria-only val          | $+4$ IoU | (checkpoint-selection sweep)                  | not committed                                                             | mixed-val                                                             | **reconfirm**       |
| Last vs best ckpt         | $\le0.3$ | —                                             | `logs/heldout/*_{best,last}.csv`                                          | best-by-val                                                           | yes (max $          |

**Notes (camera-ready actions):**

- **SDF $-3.1$:** the table row is the full-data B3 PRUE+ (ws+tta) comparison with vs without the SDF head, both on held-out-10 `object_ws_f1`: `v3_augmax_b3_sdf_full_ws_tta` (42.03) vs `repro_eval/pp_ws_tta` (45.16), $=-3.13$. The pre-augmentation direction also holds (SDF *helps* before augmentations: `b3sdf_best` 30.6 vs `b3base_best` 29.1, $+1.5$), so the "helps without augs, hurts with PRUE+" narrative is supported. (Earlier drafts quoted $-2.6$, which did not trace to a committed pair.)
- **Last vs best $\le0.3$:** across all six committed `*_best.csv` vs `*_last.csv` pairs the max $|$diff$|$ is 0.28 (`b3ff`), with most $<0.1$ (`b3base` 0.01, `b3base_aug` 0.02, `b3sdf` 0.09). Earlier drafts quoted $\pm0.5$, which overstated the spread; the row now reports the committed bound.
