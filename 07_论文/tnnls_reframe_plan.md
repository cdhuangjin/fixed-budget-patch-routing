# 027 Paper Reframe for TNNLS Submission

## Title (Updated)
**Conformal Gatekeeper: Risk-Triggered Selective Inference with Finite-Sample FPR Guarantees for Industrial Anomaly Detection**

## Target: IEEE Transactions on Neural Networks and Learning Systems (TNNLS), IF ~10.4

## Key Changes from EAAI v2

### 1. Stronger Title
- Old: "Risk-Aware Selective Inference for Resource-Constrained Industrial Anomaly Detection"
- New: Emphasizes "Conformal Gatekeeper" innovation and "Finite-Sample FPR Guarantees"

### 2. Innovation Upgrade
**Before**: Risk-aware routing (incremental contribution)
**After**: Conformal Gatekeeper (novel framework with theoretical guarantees)

Key innovation: First to provide finite-sample FPR control for selective inference in anomaly detection, inspired by NeurIPS 2025 Gatekeeper but with conformal guarantees.

### 3. Theoretical Contribution
- **Theorem 1**: Finite-sample FPR control: P(FPR <= alpha) >= 1 - delta
- **Theorem 2**: Group-conditional control via Bonferroni correction
- **Theorem 3**: Budget-constrained optimality: risk-aware selection minimizes expected loss under fixed budget

### 4. Experimental Strengths
- 3 datasets: MVTec AD (15 cats), VisA (12 cats), MPDD (6 cats) = 33 categories
- 3 seeds per dataset, 99 total category-seed units
- Budget sensitivity: 10%, 25%, 40% all evaluated
- Conformal FPR analysis: per-category thresholds
- Latency analysis: P50/P95 for fast/full/mixed paths

### 5. Results Summary

| Dataset | Risk AUROC | Random AUROC | Delta | p-value | Cohen d |
|---------|:---:|:---:|:---:|:---:|:---:|
| MVTec (45 rows) | 0.866 | 0.786 | +0.080 | 0.0002 | 0.63 |
| MPDD (18 rows) | 0.863 | 0.785 | +0.078 | 0.002 | 0.64 |
| VisA (36 rows) | 0.966 | 0.947 | +0.019 | 0.0009 | 0.70 |

### 6. Comparison with Gatekeeper (NeurIPS 2025)

| Aspect | Gatekeeper | Conformal Gatekeeper (Ours) |
|--------|:---:|:---:|
| Deferral rule | Learned via training | Conformal quantile |
| Statistical guarantee | None | Finite-sample FPR |
| Calibration | Post-hoc | Built-in |
| Group control | No | Yes (Bonferroni) |
| Domain | General ML cascades | Industrial anomaly detection |
| Evaluation | Multiple datasets | 3 industrial benchmarks |

### 7. Paper Structure (TNNLS format)

1. **Introduction** (1.5 pages)
   - Industrial anomaly detection motivation
   - Selective inference problem
   - Conformal Gatekeeper contribution

2. **Related Work** (1.5 pages)
   - Industrial anomaly detection
   - Selective prediction
   - Conformal prediction

3. **Method** (3 pages)
   - Problem formulation
   - Fast global path
   - Local patch-memory path
   - Conformal Gatekeeper routing
   - Theoretical guarantees

4. **Experiments** (4 pages)
   - Setup (datasets, metrics, baselines)
   - Main results (3 datasets)
   - Budget sensitivity analysis
   - Conformal FPR analysis
   - Latency analysis
   - Ablation studies

5. **Analysis** (1.5 pages)
   - When does risk-aware routing help?
   - Failure modes
   - Practical considerations

6. **Conclusion** (0.5 pages)

### 8. Action Items

- [x] Manuscript v2 (EAAI format) - complete
- [x] Innovation upgrade doc - complete
- [x] Stats report (99 rows, 3 datasets) - complete
- [ ] Update title and abstract for TNNLS
- [ ] Add theoretical proofs section
- [ ] Strengthen related work with Gatekeeper comparison
- [ ] Reformat to TNNLS LaTeX template
- [ ] Add more ablation studies if needed

## Estimated Timeline
- Paper reframe: 1-2 days
- LaTeX formatting: 1 day
- Final review: 1 day
- Total: 3-4 days to submission-ready
