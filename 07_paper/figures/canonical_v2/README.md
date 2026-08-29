# Canonical V2 figures

These figures are generated only by `../../02_代码/make_canonical_v2_figures.py` from the materialized Canonical V2 tables.

| Figure | Evidence source | Claim boundary |
|---|---|---|
| `fig_canonical_v2_allocation_effect.*` | `../../../05_运行记录/canonical_v2/stats_results.csv` | Paired Risk-minus-matched-Random AUROC effect at the offline strict 25% quota. Error bars are fixed-seed 95% paired-bootstrap intervals over category-seed rows. |
| `fig_canonical_v2_latency_audit.*` | `../../../05_运行记录/canonical_v2/efficiency_results.csv` | Separate MVTec batch-one CUDA system audit; not paired with the accuracy rows and not a speed claim. |

Each figure is exported as SVG, PDF, TIFF (600 dpi), and PNG (300 dpi preview). Source preflight, PDF glyph-size audit, and rendered collision audit passed after the final render.
