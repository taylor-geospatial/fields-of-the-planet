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

| Float | Generator | Source CSV(s) |
|---|---|---|
| `tab:polygon_metrics` | `make_polygon_metrics.py` | `logs/polygon_metrics/{delineate_anything_conf0005,s2_b3_augmax_full_upsampled_22,s2_upsampled_b7_augmax_full_22}.csv` (+ `*_native256.csv` for boundary); `logs/repro_eval/polygon_metrics.csv` (FTP B3) |
| `tab:area_bins` | `make_area_bins_table.py` | `logs/area_bins/{planet_b3,s2_b7,s2_b3}.csv.bins.csv` |
| `tab:heldout` | `make_heldout_results.py` | `logs/postproc_ablation/s2_{b3,b7}_augmax_full_upsampled_{nows_notta,nows_tta,ws_notta,ws_tta}.csv`; `logs/repro_eval/pp_*.csv` (FTP) |
| `tab:heldout_pc` | `make_heldout_per_country.py` | `logs/postproc_ablation/s2_b3_augmax_full_upsampled_ws_tta.csv`; `logs/repro_eval/pp_ws_tta.csv` |
| `tab:full_data` | `make_full_data_compare.py` | `logs/repro_eval/pp_ws_tta.csv` (FTP); released PRUE values from \cite{muhawenayo2026prue} |

### Planet B3 eval-run note

The dense-10 headline (`tab:polygon_metrics`, `tab:area_bins`, etc.) reads the
**most recent** Planet B3 eval, `logs/repro_eval/` (2026-06-13), giving PQ
`36.0`. The 23-region per-region breakdown (`fig:per_country_bars`,
`fig:smallholder_scatter`, per-region head-to-head prose) reads
`logs/polygon_metrics/planet_b3_augmax_full_22.csv` (2026-05-21), the only run
with all 23 regions; its dense-10 subset is PQ `35.6`. The two runs use the same
`planet_efnet3_augmax_full_best.ckpt` and differ by `<=0.4` PQ on shared
countries (eval-time settings). `35.6` is never quoted in the paper, so no
reader-visible number depends on the difference; the 10-country (`36.0`) and
23-region (`37.9`) macros are different country sets by construction.

## Script-generated figures

| Float | Generator | Source |
|---|---|---|
| `fig:aug_ablation` | `make_aug_ablation_paper.py` | `paper/scripts/output/aug_ablation_heldout10.csv` (Kenya-excluded, matching the paper's supervised-macro protocol; from `logs/heldout/*.csv`, `logs/repro_eval/pp_ws_tta.csv`). Bars 1--4 are CC-BY-subset models on the 10 held-out (OOD) countries; the final bar is the in-distribution full-data model. |
| `fig:per_country_bars` | `make_per_country_bars_pq.py` | per-country PQ from the `tab:polygon_metrics` CSVs |
| `fig:per_country_objf1_appx` | `make_per_country_both.py` | polygon PQ CSVs + released PRUE per-region Obj-F1 |
| `fig:smallholder_scatter` | `make_smallholder_scatter_paper.py` | `paper/scripts/output/smallholder_scatter.csv`; `logs/repro_eval/polygon_metrics_22.csv`; `logs/polygon_metrics/s2_upsampled_b7_augmax_full_22.csv` |
| `fig:improvement` | `make_improvement_figure.py` | `logs/per_patch/{planet_b3,s2_b7}.csv` |
| `fig:metric_example` | `make_metric_example.py` | per-patch vectorized predictions (FTP-PRUE+) |
| `fig:qualitative`, `*_appx` | `make_qualitative_v8.py`, `make_qualitative_v6/v7.py` | held-out patch predictions (both sensors) |
| `fig:qualitative_delineate` | `make_qualitative_delineate.py` | DelineateAnything YOLO11x-seg off-the-shelf predictions |

## Hand-entered tables (no generator -- values from dataset-build statistics)

These are typed directly in `main.tex`; a comment above each points to its
source. Verify against the source before camera-ready.

| Float | Source |
|---|---|
| `tab:scope` | dataset build manifest (`data/_global/` index: per-region patch/window counts, success rates) |
| `tab:udm2` | per-band UDM2 percentile statistics over the 133k UDM2 release |
| `tab:upsampled_s2_main`, `tab:upsampled_s2` | the upsampled-S2 control eval: native-256, upsample-512-at-eval, and trained-upsampled-512 rows from `logs/polygon_metrics/` + `logs/repro_eval/`; mirror each other |
