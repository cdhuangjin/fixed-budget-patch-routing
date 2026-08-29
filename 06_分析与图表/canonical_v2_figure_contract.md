# Canonical V2 figure contract

- Core conclusion: Under the offline strict 25% quota, risk routing improves AUROC relative to an equal-count random allocation; the separately audited batch-one path does not demonstrate latency reduction.
- Results question: Is the observed effect allocation quality at a fixed expensive-evaluation count, and what batch-one deployment cost accompanies the route?
- Archetype: quantitative grid with two separate figures, rather than a combined efficiency--accuracy Pareto plot.
- Backend: Python/matplotlib.
- Output: vector PDF/SVG plus 600-dpi TIFF; target width 89 mm per figure.
- Figure 1: hero = paired Risk minus matched Random mean AUROC by dataset with 95% paired-bootstrap CI; labels show category--seed rows.
- Figure 2: separate MVTec system-audit plot = batch-one mean and P95 path latency; it labels 6000 images, 15 categories, 25% fallback, CUDA synchronization, and 20 repetitions. It is not paired with Figure 1's accuracy rows.
- Evidence source: only `../05_运行记录/canonical_v2/stats_results.csv` and `../05_运行记录/canonical_v2/efficiency_results.csv`.
- Reviewer risks: bootstrap rows are not independent datasets; latency is one MVTec system audit, not a speed claim; no cross-dataset latency generalization.
