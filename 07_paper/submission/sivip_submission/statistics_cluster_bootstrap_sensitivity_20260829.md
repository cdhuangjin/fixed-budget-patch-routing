# Category-clustered bootstrap sensitivity audit

Date: 2026-08-29
Status: diagnostic only; manuscript values and narrative were not changed.

## Question

The canonical analysis bootstraps 99 category--seed rows. Because the three
seeds for one dataset category reuse that category and test set, this audit
checks whether the conclusions persist when the resampling unit is the dataset
category rather than the category--seed row.

## Method

For each dataset, the paired AUROC difference was first averaged over seeds 5,
17, and 29 within each category. The resulting category means were resampled
with replacement 10,000 times using NumPy's default random generator with seed
17. The reported intervals are the 2.5th and 97.5th percentiles of the bootstrap
mean distribution. The combined diagnostic treats each dataset--category pair
as one cluster (33 clusters total).

This is a sensitivity analysis, not a replacement inferential specification.
It does not model benchmark selection as random sampling from all industrial
domains.

## Results

| Dataset | Categories | Comparison | Mean AUROC difference | Category-clustered 95% interval |
|---|---:|---|---:|---:|
| MVTec AD | 15 | Risk - Random | +0.1249 | [0.0600, 0.1917] |
| MVTec AD | 15 | Risk - Fast-score | +0.0425 | [-0.0216, 0.1128] |
| MVTec AD | 15 | Risk - Uncertainty | +0.0746 | [0.0237, 0.1257] |
| MPDD | 6 | Risk - Random | +0.1166 | [0.0493, 0.1875] |
| MPDD | 6 | Risk - Fast-score | -0.0633 | [-0.1039, -0.0216] |
| MPDD | 6 | Risk - Uncertainty | +0.1881 | [0.0342, 0.3929] |
| VisA | 12 | Risk - Random | +0.0128 | [0.0050, 0.0201] |
| VisA | 12 | Risk - Fast-score | -0.0169 | [-0.0247, -0.0093] |
| VisA | 12 | Risk - Uncertainty | +0.0063 | [-0.0009, 0.0134] |
| Combined | 33 | Risk - Random | +0.0826 | [0.0457, 0.1217] |
| Combined | 33 | Risk - Fast-score | +0.0017 | [-0.0311, 0.0392] |
| Combined | 33 | Risk - Uncertainty | +0.0704 | [0.0282, 0.1225] |

## Interpretation boundary

- The central Risk-versus-Random allocation result remains positive for every
  dataset and for the combined diagnostic after clustering by category.
- The MVTec Risk-versus-Fast-score interval now includes zero, reinforcing the
  manuscript's cautious description of a near match rather than stable
  superiority.
- The VisA Risk-versus-Uncertainty interval also includes zero under this more
  conservative resampling unit. Any future statistical revision should avoid
  claiming a robust VisA advantage over uncertainty unless the inferential unit
  and model are justified.
- These results should be reviewed by the authors before any manuscript-level
  statistical replacement. No reported interval in the current manuscript was
  silently altered by this audit.
