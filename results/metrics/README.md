# FTP consolidated results bundle

Standalone bundle of every field-boundary metric reported in the *Fields of the Planet* (FTP)
paper, traced back to its source file. Hand this folder to anyone who asks for "the results".

## Files

- **`all_polygon_metrics_long.csv`** — tidy long format, one row per (source_file, model, country).
    Gathered from all 20 `logs/polygon_metrics/*.csv`. Columns:
    `source_file, model, imagery, backbone, license_split, protocol, country, n_patches, pq, sq, rq, f1_5_95, poly_count_delta_mean, poly_count_delta_p50, bnd_err_mean_m, bnd_err_p95_m`.
    - `rq` (== object F1 @ IoU 0.5 == RQ), `sq` (segmentation quality), `f1_5_95` (object F1 averaged
        over IoU {.5:.95}, source column `ap_5_95`).
    - `protocol`: `heldout_11country` (the CC-BY / full held-out split, 10 dense + Kenya),
        `full_data_23region` / `full_data_13region` (FTW full-data per-region protocol),
        `full_data_25region` (DelineateAnything zero-shot over all regions).
- **`macro_summary.csv`** — per-model macro-average over the **10 dense held-out countries**
    (Kenya excluded; matches `paper/figs/polygon_metrics.tex`). One row per model. The last row,
    `PRUE-FTP-B3-augmax-WS+TTA(paper-headline)`, is the watershed-postprocessed main model and is the
    authoritative source for the bold headline row (see note below).

## The 10 dense held-out countries

`belgium, cambodia, croatia, germany, latvia, lithuania, portugal, slovenia, south_africa, sweden`
(Kenya is presence-only and excluded from all supervised macros.)

## IMPORTANT — the main model row has two sources

For **every model except the headline PRUE-FTP-B3 (full)**, the paper macro equals the macro over the
matching `logs/polygon_metrics/*.csv` rows exactly.

The bold **PRUE-FTP-B3 (full)** row in `polygon_metrics.tex` / `full_data_compare.tex` /
`heldout_results.tex` / `heldout_per_country.tex` does **NOT** come from `planet_b3_augmax_full.csv`.
It comes from the watershed-postprocessed eval in **`logs/repro_eval_109588.out`** (WS+TTA polygon
block): PQ 0.360, SQ 0.779, RQ 0.452, F1 0.291, bnd-mean 7.39 m. The CSV (`planet_b3_augmax_full.csv`,
no-watershed) gives the slightly lower PQ 0.356 / RQ 0.446. The +0.006 RQ gain is the watershed
post-processing step (`obj_ws_F1` in the sweep). Both are kept in `macro_summary.csv` so the
distinction is auditable; the `(paper-headline)` row is what the paper prints.

## Paper table / number -> source map

| Paper location                                                                                      | Source                                                                                                                   |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `polygon_metrics.tex` (Tab. 1, main) — all rows except FTP-B3-full                                  | macro over `logs/polygon_metrics/<model>.csv` (= `macro_summary.csv`)                                                    |
| `polygon_metrics.tex` — **FTP-B3-full bold row**                                                    | `logs/repro_eval_109588.out` WS+TTA polygon block (PQ 0.360/SQ 0.779/RQ 0.452/F1 0.291/bnd 7.39)                         |
| `full_data_compare.tex` "Ours" Obj F1 column                                                        | `rq` from `macro_summary.csv` (B3-full uses 0.452 headline)                                                              |
| `heldout_results.tex` WS+TTA column                                                                 | `rq` from `macro_summary.csv`; FTP-B3-full = 0.452 from `repro_eval_109588.out` sweep (4-combo: 0.435/0.439/0.441/0.452) |
| `heldout_results.tex` Pix IoU column (Ours rows)                                                    | `logs/repro_eval_109588.out` 4-combo sweep `iou=` (B3-full macro 0.581 / 0.688)                                          |
| `heldout_per_country.tex` PRUE-FTP-B3(full) IoU + ObjF1(WS+TTA) cols                                | `logs/repro_eval_109588.out` block4 (`iou=`, `obj_ws_F1=`), macro IoU 0.688 / ObjF1 0.452                                |
| `heldout_per_country.tex` S2 PRUE-B3 CC-BY IoU col (macro 0.552)                                    | UNVERIFIED — not found in any provided `.out`/CSV; likely a released-checkpoint pixel-IoU eval                           |
| Tab. `tab:upsampled_s2` (appendix)                                                                  | `logs/s2up_eval_111563.out` + `s2_upsampled_b3_augmax_full.csv` (PQ 0.315/SQ 0.775/RQ 0.395/F1 0.255/bnd 28.0/p95 90.0)  |
| Tab. `tab:gsd_controlled` (appendix)                                                                | `logs/gsd_90630.out` (Planet 0.678/0.677, S2 0.654/0.630)                                                                |
| DelineateAnything row (Tab. 1)                                                                      | `delineate_anything_conf0005.csv` (PQ 0.146/SQ 0.740/RQ 0.195/F1 0.110/bnd 12.21/p95 35.60)                              |
| §Per-region head-to-head (full_data)                                                                | `planet_b3_augmax_full_22.csv` vs `s2_b7_augmax_full_22.csv` (23 regions)                                                |
| Dataset counts (Tab. scope: 70,484 patches / 140,968 targets / 6,113 scenes / 23.0 / 133,168 pairs) | `docs/planet-api-issues.md`; pairs also `docs/DATASET.md`                                                                |
| Pipeline cost (Tab. pipeline_cost: ~17h / ~84 node-h)                                               | `docs/profiling.md`                                                                                                      |
| UDM2 stats (86.6% usable, per-country rates)                                                        | release index / UDM2 CSV — NOT in this bundle's scope                                                                    |
| Smallholder scatter (Pearson r=+0.10, Spearman 0.11, n=22)                                          | per-region field-area data — NOT in this bundle's scope                                                                  |

## Regeneration

`macro_summary.csv` and `all_polygon_metrics_long.csv` are produced from `logs/polygon_metrics/*.csv`
by averaging the 10 dense-country rows per file (boundary-error columns skip Portugal's `nan` under
the CC-BY configs). The `(paper-headline)` row is copied from `logs/repro_eval_109588.out`.
