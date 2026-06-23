# Codex Pre-Submission Review: FTP / WACV 2027

Review date: 2026-06-21  
Scope: adversarial pre-submission review of `paper/main.tex`, `paper/scripts/`, `paper/refs.bib`, generated tables, eval scripts, and selected repo docs/configs.  
Rubric loaded from `/u/isaaccorley/.claude/skills/paper-writing/`: `SKILL.md`, `references/checklists.md`, `references/style-anti-patterns.md`, `references/revision-lessons.md`.

## Verdict

Not submission-ready yet.

The core result is plausible and many headline numbers do trace to CSVs, but the paper currently has three submission-blocking risks:

1. The recipe/augmentation/post-processing choices appear to be selected on the held-out test set, or at least the paper does not prove otherwise.
2. The significance/geographic-generalization paragraph is statistically indefensible for the PRUE protocol.
3. Review-time reproducibility is not available: the abstract says code/data/access instructions arrive after acceptance, while many canonical sources live in local `logs/` paths.

Top 3 fixes first:

1. Lock the final recipe using validation only, then re-evaluate once on untouched PRUE test chips; rewrite the paper to distinguish validation sweeps from final test.
2. Remove or replace the sign-test/significance claim; report single-seed per-region results descriptively unless you add seed replicates or a valid paired uncertainty analysis.
3. Provide an anonymized review artifact: code, table-generation manifest, configs, canonical CSVs, data-access instructions, license file, and exact commands.

Finding counts: CRITICAL 3 / IMPORTANT 13 / MINOR 8.

## What Checks Passed

- The paper correctly states the PRUE protocol as per-region train/validation/test chip splits, not country holdout: `paper/main.tex:174-180`.
- The headline dense-held-out polygon numbers in `tab:polygon_metrics` match the generator and CSVs to one decimal:
  - FTP B3: PQ 36.0, SQ 77.9, RQ 45.2, F1[.5:.95] 29.1, boundary mean 7.4.
  - S2 B3 upsampled object metrics: PQ 27.4, SQ 75.8, RQ 34.9, F1[.5:.95] 21.6.
  - S2 B3 native boundary: mean 18.6, p95 54.7.
- The 23-region PQ statement matches the CSVs: Planet wins 18/23; macro PQ is 38.3 vs 32.3; sign-test value recomputes to about 0.0106.
- The area-bin table matches `logs/area_bins/*.csv.bins.csv`: small-bin FTP PQ 18.4/RQ 26.1/AP 11.9; S2-B7 PQ 6.6/RQ 10.1/AP 3.5.
- No unresolved LaTeX references or citations were visible in the existing build log.
- License headline is present: dataset/models CC-BY-NC 4.0, code MIT at `paper/main.tex:108`.

## CRITICAL Findings

### C1. Final recipe appears tuned on the held-out test set, or the paper does not rule it out

Evidence:

- `paper/main.tex:239`: "Cumulative augmentation lift on Obj F1 (FTP-PRUE+ B3, dense held-out, WS+TTA)."
- `paper/main.tex:248-283`: augmentation choices are justified by held-out Obj F1 movements, ending with "We call this final recipe PRUE+."
- `paper/main.tex:287-296`: watershed and D4 TTA are adopted from gains reported on the same held-out setting.
- `paper/main.tex:800`: "Effect sizes on the 10-country dense-label held-out set unless noted".
- `paper/main.tex:823`: "Austria-only val vs mixed" appears only as a one-line ablation, not as the protocol that selected the final recipe.

Problem:

The paper repeatedly uses "held-out" results to select augmentations, post-processing, backbone choice, and checkpoint policy. If this is the official PRUE test split, then the final test is contaminated. If these sweeps were actually validation-only, the current text fails to say so. A reviewer can reject on this alone because the main result may be optimized on the evaluation set.

Concrete fix:

- Add a short "model selection protocol" paragraph before results:
  - which chips/countries were used for tuning,
  - which metric selected augmentations/post-processing/checkpoint,
  - which test set was touched only once.
- If the current sweeps used test chips, relabel them as exploratory and rerun the locked final recipe once on untouched test chips.
- Move test-set lever sweeps out of the causal argument, or report them as post-hoc diagnostics only.

### C2. Statistical significance / geographic-generalization claim is invalid under the stated protocol

Evidence:

- `paper/main.tex:421-430`: "Across the 23 test regions..." and "two-sided sign test..." then "All configurations are trained with a single seed; in place of seed replicates we assess significance across the 23 independent test regions, a stronger test of geographic generalization than optimization-noise variance."
- `paper/main.tex:174-180`: training uses each region's train chips and evaluates on that same region's test chips. This is not country-level or region-level holdout.

Problem:

The 23 regions are not independent samples of OOD generalization. Each region contributes training chips and test chips under PRUE. Treating region wins as independent trials ignores shared model training, shared hyperparameter tuning, shared dataset construction, and region-level correlation. The sign test is descriptive at best. The sentence "stronger test of geographic generalization" directly conflicts with the protocol and risks misleading reviewers into thinking this is country-holdout OOD.

Concrete fix:

- Delete the p-value and "significance" framing unless you add a statistically valid analysis.
- Replace with descriptive wording: "Planet wins 18/23 PRUE test regions under a single seed; this is a region-level descriptive count, not a country-holdout generalization test."
- If you keep statistics, add seed replicates and report mean +/- std, or use a paired bootstrap/permutation over predeclared units with caveats.

### C3. Review-time reproducibility is not sufficient

Evidence:

- `paper/main.tex:56`: "Code, release metadata, and dataset access instructions will be released after acceptance."
- `paper/scripts/make_polygon_metrics.py:6`: sources are `logs/polygon_metrics/<stem>.csv`.
- `paper/scripts/make_polygon_metrics.py:27-30`: main FTP row comes from `logs/repro_eval`.
- `paper/scripts/output/aug_ablation_heldout10.csv`: canonical CSV paths are absolute local paths such as `/u/isaaccorley/github/ftw-planet/logs/...`.
- `paper/main.tex:907-931`: the upsampled-S2 table is hand-written in `main.tex`, not generated from a script.

Problem:

WACV reviewers cannot verify the claims from "after acceptance" promises. The table scripts mostly read local, gitignored logs. Some paper tables are hand-coded. This violates the paper-writing reproducibility rubric and creates a reviewer trust problem even if all numbers are currently correct.

Concrete fix:

- Before submission, provide an anonymized artifact with:
  - code,
  - configs,
  - exact checkpoints or model cards,
  - canonical result CSVs,
  - generated figure/table commands,
  - a one-command rebuild script for all paper artifacts,
  - dataset access instructions compatible with Planet terms.
- Replace the abstract footnote with anonymized review-access language.
- Generate `tab:upsampled_s2`, `tab:scope`, and `tab:udm2` from scripts or include a clear provenance file.

## IMPORTANT Findings

### I1. "Identical labels" overstates the cross-sensor comparison

Evidence:

- `paper/main.tex:56`: "FTP ... with identical labels, splits, and training recipe".
- `paper/main.tex:104`: labels are rasterized onto the PlanetScope grid with a 3m boundary buffer.
- `paper/main.tex:477-478`: "Each model is scored against its own-sensor GT".
- `paper/main.tex:482-486`: S2 labels resolve fewer small fields because coarse rasterization merges parcels.

Problem:

The labels share the same source polygons, but the raster training/evaluation targets are not identical. Planet uses 3m rasterized masks; S2 uses 10m/upsampled rasterized masks. The distinction is central to the paper's argument, so "identical labels" is too strong.

Concrete fix:

Rewrite as: "same source polygons, FTW chip splits, and recipe; sensor-native rasterizations." Avoid "identical labels" except when referring to original vector polygons.

### I2. The main table row for S2 mixes upsampled object metrics with native-grid boundary metrics

Evidence:

- `paper/main.tex:207`: "S2 PQ/SQ/RQ/F1 use the upsampled (`resize_factor=2`) eval...; the boundary chamfer is reported at each model's native grid".
- `paper/scripts/make_polygon_metrics.py:32-37`: object columns and boundary columns are intentionally drawn from different CSVs.
- `paper/main.tex:219-220`: prose reports "PQ from 27.4 to 36.0 ... boundary error from 18.6m to 7.4m" as one result.

Problem:

The caption discloses the hybrid row, but the prose reads like a single consistent evaluation. A skeptical reviewer can argue the S2 row is cherry-picked: object metrics from the favorable upsampled eval, boundary from the favorable native eval. The appendix later shows upsampled S2 boundary is much worse (31.3m).

Concrete fix:

Split S2 rows in the main table:

- S2 native grid: all metrics native.
- S2 upsampled eval: all metrics upsampled.
- If you want "native boundary" as a separate fairness diagnostic, put it in a separate column or footnote, not as the row's only boundary value.

### I3. The abstract smallholder claim is too absolute given the per-region analysis

Evidence:

- `paper/main.tex:56`: "the gain is largest exactly where it should be".
- `paper/main.tex:73`: "the gains concentrate on the smallest (sub-0.5 ha) fields".
- `paper/main.tex:446-454`: per-region Delta PQ correlates positively with field size; Cambodia and Vietnam are losses/ties.
- `paper/main.tex:455-466`: field-level area bins recover the small-field relative-gain claim.

Problem:

The field-level bin evidence supports a relative small-field advantage, but the region-level evidence goes the other way. The abstract/contribution wording hides that tension. "Exactly where it should be" is a hostage to fortune.

Concrete fix:

Qualify the claim: "When fields are binned by individual parcel area, the relative gain is largest below 0.5 ha, although region-level gains are confounded by acquisition quality and country-specific failure modes."

### I4. Area-bin comparison uses different ground-truth populations across sensors

Evidence:

- `paper/main.tex:477-478`: "Each model is scored against its own-sensor GT".
- `paper/main.tex:482-485`: Planet has 37.9k small fields; S2 has 30.8k small fields.
- `paper/main.tex:487-488`: "Planet detects 26% of its small fields (vs S2's 10%)".

Problem:

The small-bin S2 and Planet denominators are not the same object set. This is partly the paper's point, but the wording "detects 26% vs 10%" can be read as a matched-object comparison. S2's merged small fields move into medium/large bins, so the per-bin comparison combines representation loss, training difficulty, and evaluation-target changes.

Concrete fix:

Add a matched-vector analysis or explicit decomposition:

- "label-grid representation loss" = original polygons merged/lost by S2 rasterization;
- "model detection conditional on representable GT" = compare on common/representable objects;
- "end-to-end sensor-native task" = current table.

At minimum, change prose to "on each sensor's native rasterized targets" whenever quoting 26% vs 10%.

### I5. "Obj F1" is overloaded across polygon and PRUE pixel-instance metrics

Evidence:

- `paper/figs/polygon_metrics.tex:7`: RQ is `F1_.5` in the polygon table.
- `paper/figs/heldout_results.tex:6-8`: "Obj F1" in the post-processing sweep comes from PRUE postprocess CSVs.
- `paper/main.tex:851`: appendix clarifies one occurrence is "FTW-official pixel-instance object F1 ... not the polygon recognition quality".
- `paper/main.tex:227-231`: compares FTP object F1 45.2 to released S2 47.0, then says the sets differ.

Problem:

The paper uses "Object F1" for at least two metrics: polygon RQ/F1@0.5 and PRUE pixel-instance object F1. Readers will mix them up, especially because both are on a 0-100 scale and both are called Obj F1.

Concrete fix:

Rename consistently:

- "Polygon RQ/F1@0.5" for `tab:polygon_metrics` and area bins.
- "PRUE pixel-instance F1" for `tab:heldout`, `tab:full_data`, and `app:per_country_objf1`.

Do not compare the two without a warning in the same sentence.

### I6. Metric threshold definition disagrees between prose and code

Evidence:

- `paper/main.tex:145-147`: true positive when IoU `\ge 0.5`.
- `paper/main.tex:592-593`: appendix says `IoU > \tau`.
- `scripts/eval/polygon_metrics_eval.py:114`: code uses `if iou > t:`.

Problem:

Exact equality is rare, but the paper promises "implementation-exact definitions" at `paper/main.tex:575-577`. The main text and appendix/code disagree.

Concrete fix:

Choose one convention. Prefer matching code by changing main text to `> 0.5`, or update code to `>=` and regenerate all metrics.

### I7. Single-seed results need more conservative framing

Evidence:

- `paper/main.tex:427-428`: "All configurations are trained with a single seed".
- `configs/prue/*`: `seed_everything: 7` appears throughout.

Problem:

The paper makes quantitative superiority claims without seed variance. Region-level sign counts do not substitute for optimization variance. The main result may still be large enough to survive, but the claim should not be framed as statistically established.

Concrete fix:

Run at least 3 seeds for the main B3 S2 and Planet rows, preferably 5. Report mean +/- std for PQ, RQ, F1[.5:.95], and boundary error. If compute blocks this, state "single seed" beside the table and avoid "significant" language.

### I8. DelineateAnything is not a fair main-table baseline

Evidence:

- `paper/main.tex:207`: DelineateAnything appears in the main polygon table.
- `paper/main.tex:956`: "off-the-shelf ... no FTW fine-tuning" and a modified inference protocol with `max_det` raised and minimum-area filter omitted.

Problem:

As an off-the-shelf qualitative stress test, DelineateAnything is useful. As a main quantitative row next to trained PRUE+ baselines, it is not a matched baseline. The modifications may be reasonable, but they also make the baseline neither canonical nor tuned equivalently.

Concrete fix:

Move DelineateAnything to the appendix unless you fine-tune/evaluate it under a predeclared protocol. In the main table, keep only matched S2/Planet trained baselines.

### I9. Qualitative figures are curated and should be labeled more defensibly

Evidence:

- `paper/main.tex:126`: main qualitative caption says "PlanetScope recovers parcels that Sentinel-2 misses".
- `paper/main.tex:491-513`: `fig:improvement` uses "the five held-out patches with the largest per-patch object-F1 improvement".
- `paper/scripts/make_metric_example.py:238`: candidate selection notes "the densest (best-looking) patches have the most parcels."

Problem:

The improvement figure is intentionally extreme. The main qualitative figure appears selected from dense smallholder regions. This is fine if labeled as illustrative, but the captions currently carry broad takeaways.

Concrete fix:

State selection criteria in every qualitative caption: "curated dense-smallholder examples" or "top-5 improvements, not representative." Add one randomly sampled qualitative panel or failure panel in the main/appendix.

### I10. Some paper tables are not generated from canonical scripts

Evidence:

- `paper/main.tex:384-409`: `tab:upsampled_s2_main` is handwritten.
- `paper/main.tex:907-931`: `tab:upsampled_s2` is handwritten.
- `paper/main.tex:709-737`: `tab:scope` is handwritten.
- `paper/main.tex:743-759`: `tab:udm2` is handwritten.
- `paper/scripts/` has no obvious generator for those exact `.tex` tables.

Problem:

The user's requested standard is that claimed numbers trace to canonical sources. Main headline tables mostly do. These handwritten tables are drift risks.

Concrete fix:

Add generator scripts or a single `paper/scripts/build_all_tables.py` that writes every numeric table. For handwritten tables, add an adjacent comment with exact source CSV/parquet and command.

### I11. Hyperparameters and compute are incomplete for reproduction

Evidence:

- `paper/main.tex:191-200`: lists architecture, loss, class weights, AdamW lr, epochs, batch size, crops, precision.
- Missing from paper: weight decay, scheduler, hardware/GPU count, wall-clock, seed, checkpoint-selection rule, early stopping policy, exact configs, train/val country lists, and whether B7 batch-size differences changed effective optimization.
- `paper/main.tex:903-905`: one control stopped at 77/100 epochs, but this is only explained in the appendix.

Problem:

The main text is not enough for reproduction. Configs exist, but reviewers need either a compact table or a reproducibility appendix pointing to exact YAMLs.

Concrete fix:

Add a reproducibility table with final config paths and missing details. Include compute and runtime.

### I12. License language needs a compatibility table

Evidence:

- `paper/main.tex:108`: "dataset and trained models are released under CC-BY-NC 4.0... Labels additionally carry the per-region FTW label licenses".

Problem:

If labels retain per-region licenses, the combined dataset may not be uniformly CC-BY-NC 4.0 unless every component is compatible. The sentence tries to cover this, but reviewers and dataset users need a precise license table.

Concrete fix:

Add a dataset-card/license appendix:

- Planet imagery-derived artifacts: CC-BY-NC 4.0 / Planet non-commercial terms.
- Model weights: CC-BY-NC 4.0.
- Code: MIT.
- Per-region labels: original FTW license fields, with filtering instructions.

Avoid implying all label components are relicensed if they are not.

### I13. Bibliography hygiene is not WACV-ready

Evidence:

- `paper/refs.bib:37-42`: Persello 2019 lacks volume/pages/doi.
- `paper/refs.bib:156-159`: Drusch 2012 uses "and others" in the author list.
- `paper/refs.bib:308-314`: PASTIS citation has DOI as arXiv, not venue DOI.
- `paper/refs.bib:316-323`: `pastishd` key points to OmniSAT; if this is meant as PASTIS-HD, the title/key/citation context are confusing.
- `paper/refs.bib:325-332`: Estes 2024 remains arXiv-only; maybe correct, but audit before submission.

Problem:

The paper-writing rubric explicitly calls out bib hygiene. This bibliography has enough incomplete or confusing entries to invite reviewer friction.

Concrete fix:

Run a DOI/Crossref/Semantic Scholar audit for all entries. Fix metadata, capitalization, author lists, and published venue fields. If `pastishd` really means a dataset introduced by OmniSAT, say that in prose or use a clearer key.

## MINOR Findings

### M1. Abstract overuses absolute language

Evidence:

- `paper/main.tex:56`: "recovers exactly the parcels Sentinel-2 loses"; "exactly where it should be".
- `paper/main.tex:63`: "no model trained at 10m can recover them".
- `paper/main.tex:901`: "direct evidence"; `paper/main.tex:900`: "unambiguous".

Problem:

These are hostages to fortune. The evidence supports "under this protocol," not universal claims.

Concrete fix:

Replace with qualified claims: "in our sensor-native rasterization/evaluation," "the strongest gains in the field-size bin analysis," "supports," "consistent with."

### M2. Captions are too long and sometimes carry arguments better left in text

Evidence:

- `paper/main.tex:207`: main table caption is a long paragraph.
- `paper/main.tex:856`: appendix figure caption argues why pixel-instance F1 understates benefit.
- `paper/main.tex:910-916`: table caption repeats interpretation from prose.

Problem:

Captions should be self-contained, but these are doing too much. Long captions hide the important caveats.

Concrete fix:

Shorten captions to one takeaway plus scale/protocol notes. Move causal interpretation to body text.

### M3. LaTeX build has appendix float warnings

Evidence:

- Existing `paper/main.log`: "Float too large for page by 12.27986pt on input line 944."
- Existing `paper/main.log`: "Float too large for page by 12.27986pt on input line 951."

Problem:

Appendix qualitative figures may overrun pages in the submitted PDF.

Concrete fix:

Render the final PDF and reduce the offending appendix figure heights/widths or split them.

### M4. Lowercase country names in captions look unpolished

Evidence:

- `paper/main.tex:873-875`: "netherlands", "lithuania", "cambodia", "vietnam", "germany".
- `paper/main.tex:800`: "presence-only kenya".

Problem:

Small but visible polish issue.

Concrete fix:

Capitalize country names in prose/captions.

### M5. Appendix prose repeats "presence-only Kenya" many times

Evidence:

- `paper/main.tex:180-186`, `551`, `659`, `774-783`, `838-843`.

Problem:

The caveat is important, but repetition consumes space and distracts.

Concrete fix:

Define once in evaluation protocol and use short parentheticals afterward.

### M6. Stale source comments should be removed before camera-ready source/supplement submission

Evidence:

- `paper/main.tex:216-218`: "Rebuild 2026-06-21..."
- `paper/scripts/make_smallholder_scatter_paper.py:59-62`: "Earlier this read the stale pq columns..."

Problem:

PDF reviewers will not see comments, but source/supplement reviewers may. Stale-fix comments suggest result churn.

Concrete fix:

Keep provenance in a changelog or review note, not in final source comments.

### M7. Paper scripts include broad exception swallowing in checkpoint loading

Evidence:

- `scripts/eval/polygon_metrics_eval.py:393-410`: broad `except Exception` when trying task classes.

Problem:

This is not a paper-text issue, but it affects reproducibility. A real checkpoint-loading bug could be swallowed and misdiagnosed.

Concrete fix:

Catch expected Lightning class mismatch exceptions narrowly, or log the exception before falling through.

### M8. UDM2 "usable" threshold should be connected to final yield more clearly

Evidence:

- `paper/main.tex:113`: "86.6% of patches are usable at a strict threshold".
- `paper/main.tex:541`: final release yield is 94.5% of (patch, window) targets.

Problem:

These are different concepts: UDM2 quality usability vs acquisition/yield. Readers may confuse them.

Concrete fix:

Add one sentence: "The 86.6% statistic is a per-image quality flag after pairing; it is not the same denominator as the 94.5% acquisition yield."

## Claims vs Evidence Summary

| Claim | Evidence status | Fix |
|---|---|---|
| FTP improves PQ 27.4 -> 36.0 and boundary error 18.6m -> 7.4m | Numerically traces to CSVs, but S2 row mixes upsampled object metrics with native boundary | Split native/upsampled rows or mark hybrid more explicitly |
| FTP nearly triples small-field PQ/F1 | Area-bin numbers trace, but denominators differ by sensor-native GT | Qualify as sensor-native end-to-end; add common-object decomposition |
| Gains concentrate on smallest fields | Supported by field-level relative bins; contradicted by per-region trend | Rewrite abstract/contribution to mention both |
| 18/23 per-region PQ wins are significant/geographic | Not defensible under PRUE train/test protocol and single seed | Remove p-value/significance claim |
| Matched architecture/recipe isolates imagery | Mostly for B3 main comparison, but recipe appears selected on held-out test | Document validation-only selection or rerun locked final recipe |
| Dataset/models CC-BY-NC, code MIT | Stated, but label-license compatibility needs detail | Add license table / dataset card |
| Every artifact rebuilds from canonical source | Partly true for generated tables; false/unclear for hand-written tables and local logs | Add build manifest and generate all numeric tables |

## Final Recommendation

The paper has a strong, publishable core: FTP is useful, the polygon-first framing is right for field boundaries, and the main numeric deltas mostly check out. Do not submit with the current statistical and model-selection language. Fix protocol defensibility first, then reproducibility, then polish wording/figures.
