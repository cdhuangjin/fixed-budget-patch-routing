# 027 Innovation Upgrade: Conformal Gatekeeper for Industrial Cascades

## Inspiration: Gatekeeper (NeurIPS 2025)
Gatekeeper trains small models to confidently handle easy tasks and defer hard ones.
**Our twist**: Add conformal finite-sample guarantees to the deferral rule.

## Core Innovation: Conformal Gatekeeper

### Problem
- Gatekeeper: learns deferral rule via training, no statistical guarantees
- Standard conformal prediction: guarantees coverage but not deferral quality
- **Gap**: no method provides both deferral quality AND finite-sample FPR control

### Solution: Conformal Gatekeeper
1. **Risk-calibrated deferral**: Train fast model to output calibrated risk scores
2. **Conformal threshold**: Set deferral threshold via conformal quantile for FPR control
3. **Group-conditional guarantees**: Per-category FPR control via Bonferroni correction

### Mathematical Framework
Deferral rule: defer(x) = 1[S(x) >= q_{1-alpha}]
where S(x) is the risk score and q_{1-alpha} is the conformal threshold.

**Theorem**: P(FPR <= alpha) >= 1 - delta (finite-sample guarantee)

### Key Difference from Gatekeeper
| Aspect | Gatekeeper | Conformal Gatekeeper (Ours) |
|--------|-----------|---------------------------|
| Deferral rule | Learned via training | Conformal quantile |
| Guarantee | None | Finite-sample FPR |
| Calibration | Post-hoc | Built-in |
| Group control | No | Yes (Bonferroni) |

## Evidence (Already Complete)
- 3 datasets (MVTec/VisA/MPDD), 33 categories, 3 seeds
- Budget sensitivity: 10%/25%/50% all show risk > random
- Conformal FPR: group-conditional thresholds computed
- Latency: P50/P95 for fast/full/mixed paths

## Target Venue
ICML 2026 / NeurIPS 2026 (if theory is developed properly)
Or: TNNLS / IJCV (if framed as application)
