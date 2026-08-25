# Statistical and figure audit for the EAAI manuscript

## Study-design readout

- Dataset: MVTec AD, 15 categories.
- Training data: `train/good` normal images only.
- Formal seeds: 17, 29 and 41.
- Primary analysis unit: one category--seed unit, yielding 45 units for the main comparison.
- Primary endpoints: image-level AUROC and Recall@5%FPR.
- Summary convention: mean over the 45 category--seed units; these are not image-level independent replicates.
- Primary comparison: Risk versus Random with exactly matched fallback counts within each category--seed unit.
- Uncertainty: paired bootstrap resampling of the 45 category--seed units; 95% confidence intervals for Risk minus Random.
- Latency: descriptive P50/P95 summaries from repeated batch-one CUDA timing; repeated timing observations are not treated as independent experimental units.

## Major statistical issues

### [P1] Define the independent unit

The manuscript should explicitly state that `n=45` refers to category--seed units, not individual images. Confidence intervals should be described as unit-level paired bootstrap intervals.

### [P1] Treat the budget sweep as sensitivity analysis

The 10%, 25% and 50% results are from seed 17 across 15 categories. They demonstrate a descriptive budget trend, but the 10% and 50% points have only one seed. The 25% point is the formal three-seed analysis. Do not call each budget difference statistically significant.

### [P1] Clarify latency replication

Repeated latency measurements characterize runtime variability and tail latency; they should not be treated as independent samples for hypothesis testing. Report latency descriptively with batch size, warm-up count, repetition count, CUDA synchronisation and memory-bank configuration.

### [P2] Define multiplicity strategy

No p-values or star-based significance labels are used. Confidence intervals and effect sizes are the primary inferential summaries; budget, localisation and latency analyses are secondary descriptive evaluations.

### [P2] Provide per-unit source data

The per-category--seed values used to recompute the bootstrap intervals have now been exported. The summary figure data and unit-level source data should be submitted together with the analysis script.

## Ready-to-paste Statistical analysis text

Statistical analyses were performed with Python 3.12 and PyTorch 2.8.0 for model evaluation. The primary analysis unit was a category--seed unit: each of the 15 MVTec AD categories was evaluated with each of three seeds (17, 29 and 41), yielding 45 units. Image-level AUROC and Recall@5%FPR were summarised as the mean across these units. The primary comparison was local Patch routing versus matched Random with exactly matched fallback counts within each category--seed unit. Uncertainty was estimated by paired bootstrap resampling of the 45 category--seed units, and 95% confidence intervals were calculated for the route-minus-Random difference. The 10%, 25% and 50% budget sweep, patch-aggregation ablation, localisation audit and latency benchmark were treated as secondary descriptive evaluations; no p-values or star-based significance labels are reported for these analyses. Latency was summarised by P50 and P95 over repeated batch-one CUDA measurements after warm-up and explicit CUDA synchronisation; repeated timing measurements were not treated as independent experimental units. No test labels were used for threshold calibration, and no observations were excluded from the formal evaluations.

## Ready-to-paste figure legends

### Fig. 1

**Fig. 1 | Risk-aware escalation improves budgeted anomaly detection over matched random escalation.** a, Macro-mean image-level AUROC and Recall@5%FPR for Fast, Random matched fallback, Risk fallback and Full on MVTec AD. Values are means over 45 category--seed units (15 categories and three seeds). Random and Risk use the same realised fallback count. b, AUROC across target calibration budgets for Random and Risk on seed 17 and 15 categories. c, Recall@5%FPR across the same budgets. d, P50 and P95 end-to-end latency from the unified batch-one CUDA benchmark. Latency values are descriptive summaries of repeated measurements and are not treated as independent replicates. Summary source data are provided in `source_data_eaai_figures.csv`, and unit-level source data are provided in `mvtec_category_seed_unit_results.csv`.

### Fig. 2

**Fig. 2 | Paired bootstrap uncertainty for risk-aware routing.** Risk-minus-Random differences in AUROC and Recall@5%FPR. Points show the observed mean difference over 45 category--seed units; horizontal bars show 95% confidence intervals from paired bootstrap resampling of those units. The vertical dashed line marks zero. Summary source data and the per-unit values used to recompute the intervals are provided in the accompanying source-data files.

## AUTHOR_INPUT_NEEDED before final submission

1. Confirm the exact Python, NumPy, scikit-learn and CUDA versions used for the final rerun.
2. Confirm whether the final repository will include all raw timing repetitions or only the timing summary plus the benchmark script.

## Reviewer-risk note

A statistical reviewer is most likely to ask whether category--seed units, images and repeated timing measurements have been distinguished. The current wording is defensible if the unit-level definition remains explicit and if the manuscript avoids p-value or causal language for the descriptive budget and latency analyses.
