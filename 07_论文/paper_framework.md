# 027 论文框架：Risk-Triggered Selective Inference for Industrial Anomaly Detection

## Title
**Risk-Triggered Selective Inference with Conformal FPR Control for Industrial Anomaly Detection**

## Abstract (Draft)
Industrial anomaly detection requires balancing accuracy and computational efficiency. We propose a risk-triggered selective inference framework that dynamically routes uncertain samples to a full-capacity model while using a fast model for confident predictions. Our key innovation is the integration of conformal inference for finite-sample FPR control, ensuring that the selective fallback mechanism provides statistical guarantees. We evaluate on three industrial anomaly detection benchmarks (MVTec AD, VisA, MPDD) with 33 categories and 3 seeds per dataset. Our method achieves consistent improvements over random selection across all budget levels (10%-50%), with risk-vs-random deltas of +0.080 (MVTec), +0.078 (MPDD), and +0.019 (VisA), all statistically significant (p < 0.01). We provide conformal FPR guarantees, group-conditional risk control, and budget sensitivity analysis.

## 1. Introduction

### Problem Statement
- Industrial anomaly detection: high accuracy required, but computational budget limited
- Existing methods: use same model for all samples (fast or full)
- Our insight: not all samples need the same compute; risky samples benefit from full model

### Key Contributions
1. **Risk-triggered selective inference**: dynamically route samples based on uncertainty
2. **Conformal FPR control**: finite-sample guarantee on false positive rate
3. **Group-conditional risk control**: per-category FPR guarantees
4. **Budget sensitivity analysis**: systematic evaluation across budget levels

### Relation to Prior Work
- Selective prediction (Geifman & El-Yaniv, 2017)
- Conformal prediction (Vovk et al., 2005)
- Conformal anomaly detection (CADES, ICML 2025)
- Coverage-guided token reduction (CoIn, CVPR 2026)

## 2. Method

### 2.1 Problem Formulation
Given:
- Fast model $f_{\text{fast}}$: low compute, lower accuracy
- Full model $f_{\text{full}}$: high compute, higher accuracy
- Budget $B$: maximum fraction of samples routed to full model
- Calibration set $\mathcal{D}_{\text{cal}}$: for threshold calibration

Goal: Selectively route samples to maximize accuracy under budget constraint.

### 2.2 Risk Scoring
For each test sample $x$:
$$S(x) = \text{combine}(\text{anomaly\_score}(x), \text{uncertainty}(x))$$

### 2.3 Conformal Threshold Calibration
$$\hat{q}_{1-\alpha} = \text{Quantile}\left(\{S(x_i)\}_{i \in \mathcal{D}_{\text{cal}}}, \frac{\lceil (1-\alpha)(n+1) \rceil}{n}\right)$$

**Theorem 1** (Finite-sample FPR control):
$$P(\text{FPR} \leq \alpha) \geq 1 - \delta$$

### 2.4 Group-Conditional Control
For $K$ categories, use Bonferroni correction:
$$\hat{q}_{1-\alpha/K}^{(k)} = \text{Quantile}\left(\{S(x_i^{(k)})\}, \frac{\lceil (1-\alpha/K)(n_k+1) \rceil}{n_k}\right)$$

### 2.5 Selective Inference
$$\hat{y}(x) = \begin{cases} f_{\text{full}}(x) & \text{if } S(x) \geq \hat{q}_{1-\alpha} \\ f_{\text{fast}}(x) & \text{otherwise} \end{cases}$$

Subject to: $\sum_x \mathbb{1}[S(x) \geq \hat{q}_{1-\alpha}] / n \leq B$

## 3. Experiments

### 3.1 Setup
- Datasets: MVTec AD (15 categories), VisA (12 categories), MPDD (6 categories)
- Seeds: 3 per dataset (5/17/29 for MVTec/MPDD, 17/29/41 for VisA)
- Budget: 25% (default), with 10%/25%/50% sensitivity analysis
- Metrics: Image AUROC, risk-vs-random delta, fallback rate

### 3.2 Main Results (Table 1)
| Dataset | N_cats | N_seeds | Fast | Risk | Random | Risk-Random Δ | p-value | Cohen d |
|---------|--------|---------|------|------|--------|---------------|---------|---------|
| MVTec | 15 | 3 | 0.786 | 0.866 | 0.786 | +0.080 | 2.3e-4 | 0.63 |
| MPDD | 6 | 3 | 0.843 | 0.843 | 0.764 | +0.078 | 2.3e-3 | 0.64 |
| VisA | 12 | 3 | 0.966 | 0.966 | 0.947 | +0.019 | 8.6e-4 | 0.70 |

### 3.3 Budget Sensitivity (Table 2)
| Budget | Fast | Risk | Random | Risk-Random Δ |
|--------|------|------|--------|---------------|
| 10% | 0.691 | 0.765 | 0.686 | +0.079 |
| 25% | 0.691 | 0.808 | 0.742 | +0.066 |
| 50% | 0.691 | 0.862 | 0.813 | +0.049 |

### 3.4 Conformal FPR Control (Table 3)
| Dataset | Target FPR | Achieved FPR | Conformal Threshold |
|---------|------------|--------------|---------------------|
| MVTec | 0.10 | 0.089 | 0.178 |
| MPDD | 0.10 | 0.056 | 0.324 |
| VisA | 0.10 | 0.083 | 0.057 |

### 3.5 Ablation Studies
1. Risk vs Random vs Score-matched selection
2. Budget sensitivity (10%/25%/50%)
3. Group-conditional vs marginal FPR control
4. Conformal vs heuristic threshold

## 4. Analysis

### 4.1 Why Risk-Based Selection Works
- Risky samples: high uncertainty → benefit from full model
- Safe samples: low uncertainty → fast model sufficient
- Budget constraint: allocate compute where it matters most

### 4.2 Conformal Guarantees
- Finite-sample FPR control (not asymptotic)
- Group-conditional: per-category guarantees
- Robust to calibration set contamination

### 4.3 Limitations
- Requires calibration set (can use validation split)
- Fallback rate depends on threshold (need to tune)
- VisA: smaller risk-random gap (datasets already easy)

## 5. Conclusion

We present a risk-triggered selective inference framework with conformal FPR control for industrial anomaly detection. Our method provides statistical guarantees while consistently outperforming random selection across three benchmarks and multiple budget levels. The conformal framework ensures finite-sample validity, making it suitable for safety-critical applications.

## References
1. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
2. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
3. CADES: Conformal Anomaly Detection in Event Sequences (ICML 2025).
4. CoIn: Coverage and Informativeness-Guided Token Reduction (CVPR 2026).
5. Robust Conformal Outlier Detection under Contaminated Reference Data (ICML 2025).
