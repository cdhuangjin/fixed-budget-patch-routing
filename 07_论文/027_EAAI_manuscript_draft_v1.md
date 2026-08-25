# Risk-Aware Selective Inference for Resource-Constrained Industrial Anomaly Detection

## Abstract

Industrial anomaly detection systems must process many normal samples while allocating detailed analysis to the relatively small number of samples that may contain subtle defects. Applying a high-cost local detector to every image provides a strong accuracy reference but wastes computation on easy cases; random escalation controls the computation budget but does not use sample-specific risk. Here we study whether a low-cost global risk score can allocate a high-cost local patch-memory detector more effectively than random escalation under an exactly matched fallback budget. The proposed two-stage system first extracts a low-resolution global representation with an ImageNet-pretrained ResNet-18 and then escalates only high-risk samples to a higher-resolution local patch-memory path. The routing threshold is calibrated using normal images only. On the 15 categories of MVTec AD with three random seeds, risk-aware routing achieved a mean image-level AUROC of 0.8674 and Recall@5%FPR of 0.7251, compared with 0.7796 and 0.6204 for a random controller with exactly the same number of escalated samples. The paired bootstrap differences were +0.0879 (95% CI, 0.0676–0.1081) for AUROC and +0.1047 (95% CI, 0.0754–0.1337) for Recall@5%FPR. The advantage remained at target budgets of 10%, 25% and 40%. A unified batch-one CUDA benchmark showed that the selective pipeline does not reduce per-sample P95 latency relative to the full detector; its benefit is instead the preferential allocation of expensive analysis to high-risk samples. These results establish a bounded systems contribution: risk-aware escalation improves the quality of budgeted industrial anomaly detection over matched random escalation, while its applicability is limited by distribution shift, routing errors and sequential two-stage execution.

**Keywords:** industrial anomaly detection; selective inference; risk-aware routing; patch-memory detection; resource-constrained inference; MVTec AD

## 1. Introduction

Industrial visual inspection combines a large stream of normal products with a small and heterogeneous set of defective products. Most normal images are visually simple, whereas subtle texture defects, small local changes and boundary anomalies may require higher-resolution feature extraction and local comparison. A detector that applies detailed analysis to every image can provide high accuracy, but its computational cost is incurred even when the image is easy. Conversely, a uniformly lightweight detector reduces computation but may miss anomalies that are not well represented by global features.

This creates a selective inference problem. The central question is not only which detector is more accurate, but also which samples should receive additional computation. A practical system should use a low-cost path to estimate sample risk and reserve a high-cost path for samples that are likely to benefit from further analysis. The difficulty is that a reported improvement can be confounded by the number of samples sent to the high-cost detector: a controller may appear better simply because it uses more expensive computation. A fair evaluation therefore requires a control that escalates exactly the same number of samples but selects them independently of the risk score.

In this work, we develop and evaluate a risk-aware two-stage anomaly detection system. The fast path uses low-resolution global features to produce a normality-based risk score. Samples above a threshold calibrated on normal images are escalated to a high-resolution local patch-memory detector. The final system score is the fast score for non-escalated samples and the local score for escalated samples. We compare this system with a random controller whose fallback count is matched exactly to the risk-aware system for every category and seed. This comparison isolates the value of selecting *which* samples receive expensive analysis.

Our main result is that risk-aware escalation consistently outperforms matched random escalation on MVTec AD. Across 45 category–seed units, the risk-aware controller improved AUROC in 35 units and Recall@5%FPR in 36 units. The pooled mean AUROC was 0.8674 for risk-aware routing and 0.7796 for matched random routing. The corresponding recall values were 0.7251 and 0.6204. The advantage was observed at three budget settings, indicating that it was not caused by a single threshold choice.

The contribution is deliberately bounded. The proposed system does not replace the full detector, and it does not claim to reduce end-to-end single-sample P95 latency. In the measured sequential implementation, the risk-aware pipeline had a P95 latency of 16.47 ms, compared with 10.52 ms for the full path. The contribution is therefore a budget–quality allocation mechanism: under a fixed number of expensive fallbacks, risk-based selection produces better detection performance than random selection.

The contributions of this study are:

1. a two-stage industrial anomaly detection pipeline that combines a low-cost global risk scorer with a high-cost local patch-memory detector;
2. an exactly matched random-escalation protocol that separates the value of risk-based selection from the value of using additional computation;
3. a multi-seed and multi-budget evaluation covering image-level detection, localising capability of the full path, and unified CUDA latency; and
4. an explicit analysis of failure modes, including routing errors, normal-distribution shift and the latency cost of sequential execution.

## 2. Related work

### 2.1 Industrial visual anomaly detection

Industrial anomaly detection is commonly formulated as learning the distribution of normal samples and identifying deviations at test time. Feature-memory methods compare test representations with stored normal representations and have become effective because they can use normal-only training data while retaining local spatial information [CITATION NEEDED: primary industrial anomaly detection methods]. Patch-based methods such as PatchCore-style detectors use local feature memories and nearest-neighbour distances to detect and localise defects [CITATION NEEDED: PatchCore]. Gaussian feature-distribution methods such as PaDiM model local feature statistics and provide a complementary reference based on parametric normality modelling [CITATION NEEDED: PaDiM].

These methods typically define one inference path for every image. Their accuracy and computational cost are consequently coupled: using a high-resolution local path for all images improves coverage of small defects but increases the cost of easy normal samples. Our work does not propose a new patch-memory detector. Instead, it studies how a fast global path can decide which inputs should receive an existing high-cost local analysis.

### 2.2 Selective prediction and conditional computation

Selective prediction allocates model computation or abstention decisions according to input difficulty, uncertainty or estimated error risk [CITATION NEEDED: selective prediction and conditional computation]. In computer vision, conditional computation has been used to activate different network components or resolutions for different inputs [CITATION NEEDED: conditional computation]. These studies motivate risk-dependent computation, but industrial anomaly detection adds two constraints. First, training often uses only normal images, so the routing signal should not require anomalous validation labels. Second, the evaluation must distinguish the benefit of risk-based selection from the benefit of simply increasing the fraction of samples processed by the expensive model.

We address these constraints with normal-only threshold calibration and an exact-count random control. The resulting evaluation treats routing as a resource allocation problem rather than as an unqualified accuracy comparison between two detectors.

## 3. Method

### 3.1 Problem formulation

Let (x) denote an input image. The system contains a fast detector (f_s), a local detector (f_f), and a binary routing function (r). The fast detector produces a global anomaly score (s_s(x)). The route decision is

\[
r(x)=\mathbb{1}[s_s(x)\geq \tau],
\]

where (	au) is estimated from normal calibration images. The final score is

\[
s(x)=(1-r(x))s_s(x)+r(x)s_f(x),
\]

where (s_f(x)) is the local patch-memory score. The route is therefore a selective replacement of the fast score, not an ensemble average of the two scores.

For every category and seed, the matched random controller uses the same number of escalated test images as the risk-aware controller. It samples the escalated images without replacement using a fixed seed and leaves the detector outputs unchanged. Thus, the primary comparison holds the fallback count constant and changes only the selection rule.

### 3.2 Fast global path

The fast path resizes each image to (128\times128) pixels and extracts the layer-2 feature map of an ImageNet-pretrained ResNet-18. Spatial average pooling followed by (L_2) normalisation produces a global descriptor (z_s(x)). Descriptors from normal training images form a normal memory bank \(\mathcal{M}_s\). The fast anomaly score is the nearest-neighbour distance to this bank:

\[
s_s(x)=\min_{z\in\mathcal{M}_s}\|z_s(x)-z\|_2.
\]

The fast path is used for screening and risk ranking. It is not assumed to be a complete replacement for high-resolution local analysis.

### 3.3 High-cost local patch-memory path

The local path resizes each image to (224\times224) pixels and extracts layer-3 ResNet-18 patch descriptors. Each patch descriptor is normalised, and descriptors from all normal training images form a local memory bank \(\mathcal{M}_f\). For each test patch, the detector computes the nearest-neighbour distance to \(\mathcal{M}_f\). The image-level score is the mean of the largest 10% of patch distances:

\[
s_f(x)=\operatorname{MeanTop}_{10\%}\left(\left\{\min_{z\in\mathcal{M}_f}\|p_i(x)-z\|_2\right\}_i\right).
\]

The same patch-distance map can be upsampled to the input resolution for pixel-level localisation. In this study, the local path is a high-cost reference detector and the contribution is the routing mechanism around it.

### 3.4 Normal-only calibration and budget control

Normal training images are divided into a memory-bank subset and an independent calibration subset. No test labels are used to choose (	au). For a target calibration budget (b), the threshold is the ((1-b))-quantile of fast scores on the normal calibration subset. Because the normal distribution may shift between calibration and test traffic, the realised fallback rate is reported explicitly rather than treated as a guaranteed budget.

### 3.5 Implementation and latency measurement

All methods use the same ImageNet-pretrained ResNet-18 and the same MVTec AD split. Formal accuracy experiments use seeds 17, 29 and 41. Latency is measured with batch size one, 20 warm-up repetitions, at least 100 timed repetitions per image, CUDA synchronisation and a common cached memory-bank configuration. End-to-end timing includes routing, feature extraction, nearest-neighbour scoring and synchronisation. This measurement is intended to describe the current sequential implementation; it is not a theoretical FLOPs estimate.

## 4. Experimental setup

### 4.1 Dataset and protocol

We use MVTec AD, which contains 15 object and texture categories with normal-only training images and defective test images [CITATION NEEDED: MVTec AD]. Training uses only `train/good` images. The official test images are used for final evaluation. Results are macro-averaged over categories. The main study contains 15 categories × 3 seeds = 45 category–seed units.

The primary metrics are image-level AUROC and Recall@5%FPR. AUROC measures ranking quality across operating points, whereas Recall@5%FPR measures anomaly retrieval under a low false-positive constraint. For paired uncertainty estimates, category–seed units are resampled with replacement to obtain 95% bootstrap confidence intervals for Risk minus Random.

### 4.2 Compared systems

We compare four systems: Fast only, which never escalates; Random matched fallback, which escalates the same number of images as Risk but chooses them randomly; Risk fallback, which escalates high-risk images selected by the fast score; and Full only, which applies the local detector to every image. We additionally report a PaDiM-style diagonal-Gaussian baseline using the same layer-3 patch features. This implementation is a controlled reference and is not claimed to reproduce every component of the official PaDiM system.

## 5. Results

### 5.1 Risk-aware selection improves detection under matched fallback counts

Risk-aware routing substantially improved performance over the matched random controller while using exactly the same number of escalated samples. The mean image-level AUROC was 0.8674 for Risk and 0.7796 for Random. Recall@5%FPR was 0.7251 and 0.6204, respectively. The difference was +0.0879 AUROC points (95% bootstrap CI, 0.0676–0.1081) and +0.1047 recall points (95% bootstrap CI, 0.0754–0.1337). Risk improved AUROC in 35 of 45 category–seed units and recall in 36 of 45 units. Fallback counts matched exactly in all 45 units.

| Method | Image AUROC | Recall@5%FPR | Actual fallback rate |
|---|---:|---:|---:|
| Fast only | 0.7860 | 0.4575 | 0.0000 |
| Random matched fallback | 0.7796 | 0.6204 | 0.7197 |
| Risk fallback | **0.8674** | **0.7251** | **0.7197** |
| Full only | 0.9390 | 0.7852 | 1.0000 |

Full remained the strongest accuracy reference. This comparison is important for interpreting the contribution: Risk did not surpass Full, but it used a risk-based rule to obtain a better result than random escalation at the same realised fallback count.

The PaDiM-style diagonal-Gaussian reference achieved a mean AUROC of 0.8982 and Recall@5%FPR of 0.6544 over the same 45 category–seed units. The score-ranked matched ablation was numerically identical to Risk, because the current Risk implementation ranks samples by the fast anomaly score and selects the required number of highest-risk samples. An oracle test-label upper bound achieved AUROC 0.9718 and Recall@5%FPR 0.9257; this uses test labels and is reported only as an analytical upper bound, not as a deployable method.

### 5.2 The advantage persists across budgets

Risk-aware selection remained better than matched random selection at all three evaluated target budgets. At a 10% target budget, Risk achieved AUROC 0.8322 and Recall@5%FPR 0.6404, compared with 0.7379 and 0.5222 for Random. At 25%, the corresponding values were 0.8652 and 0.7269 versus 0.7889 and 0.6396. At 40%, Risk achieved 0.8886 and 0.7512 versus 0.8264 and 0.7035.

| Target budget | Random AUROC | Risk AUROC | Random Recall | Risk Recall | Risk realised fallback |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.7379 | **0.8322** | 0.5222 | **0.6404** | 0.5757 |
| 25% | 0.7889 | **0.8652** | 0.6396 | **0.7269** | 0.7223 |
| 40% | 0.8264 | **0.8886** | 0.7035 | **0.7512** | 0.8217 |

The realised fallback rates were higher than the nominal calibration targets. This difference reflects the shift between the normal calibration distribution and the normal test distribution. It means that the budget values should be interpreted as calibration targets rather than strict deployment guarantees.

### 5.3 Localisation capability and latency boundary

The full local path achieved a mean pixel-level AUROC of 0.9543 and a mean PRO score of 0.8380 at FPR 0.30 over the 15 categories in the localisation audit. These measurements establish the spatial detection capability of the local path. They do not establish that every sample routed by Risk receives a localisation map, because non-escalated samples retain the fast-path output.

In the unified batch-one CUDA benchmark, Fast, Full and Risk achieved P50/P95 latencies of 2.44/7.35 ms, 4.09/10.52 ms and 5.85/16.47 ms, respectively. The Risk pipeline was slower in end-to-end P95 because it first computes the fast path and then executes the local path for selected samples. Therefore, the current implementation does not support a claim of lower per-sample P95 latency. Its demonstrated value is selective allocation of expensive analysis, not direct tail-latency reduction.

### 5.4 Failure modes

Risk-aware routing did not improve every category–seed unit. The failure cases are consistent with a limitation of global screening: a local defect may produce a weak global risk signal and therefore avoid escalation. In addition, normal-distribution shift can increase the realised fallback rate beyond the calibration target. These failure modes indicate that routing quality and budget stability are separate properties. Improving one does not automatically guarantee the other.

## 6. Discussion

This study shows that the selection rule can matter as much as the fallback count in a two-stage anomaly detector. When the number of expensive local evaluations is held fixed, selecting samples by a fast normality-based risk score is more effective than selecting them randomly. The result is supported by exact per-unit matching, three random seeds, paired bootstrap intervals and a three-point budget sweep.

The evidence also clarifies what the system does not solve. The Full detector remains more accurate because every image receives local analysis. The sequential Risk implementation is not faster in single-sample P95 because routing requires the fast computation before the local computation. A production deployment would therefore need a throughput-oriented batching strategy, asynchronous scheduling, a calibrated budget controller, or a genuinely cheaper local kernel before claiming an end-to-end latency advantage.

The method's current novelty is best understood as a system-level resource allocation mechanism around a normal-memory anomaly detector. It is not a new backbone, a new PatchCore formulation or a learned uncertainty model. This positioning is useful for industrial AI applications because it makes the evaluation question explicit: under a fixed high-cost analysis budget, does risk-based selection improve the quality of the processed stream?

Several limitations remain. First, the main evidence comes from MVTec AD and should not be generalized to all industrial environments. Second, the current Risk route uses the fast anomaly score directly; the score-ranked ablation therefore does not constitute an independent learned router. Third, the localisation results characterize the Full local path rather than the coverage and quality of localisation delivered by the selective system. Finally, no external industrial dataset was available in the current study. These limitations motivate external-data validation, route-specific localisation coverage, adaptive budget calibration and throughput experiments as the next steps.

## 7. Conclusion

We presented a risk-aware selective inference system for industrial anomaly detection. The system uses a fast global path to rank sample risk and assigns a high-cost local patch-memory analysis to selected samples. On MVTec AD, Risk-aware routing outperformed an exactly matched random controller across image-level AUROC, Recall@5%FPR and three budget settings. The evidence supports a specific conclusion: risk-based allocation of expensive anomaly analysis is more effective than random allocation under the same realised fallback count. The evidence does not support claims that the system surpasses the Full detector or reduces sequential single-sample P95 latency. This bounded result provides a reproducible basis for resource-aware industrial anomaly detection while defining the conditions under which further engineering and external validation are required.

## Data and code availability

The study uses the publicly available MVTec AD dataset. The exact data split, calibration protocol, route seeds, result files and evaluation scripts should be deposited in a public repository before submission. **[AUTHOR ACTION: insert repository URL, software version and dataset access statement.]**

## Declaration of competing interests

The authors declare no competing interests.

## References to complete before submission

The following citation classes must be replaced by verified primary references:

1. MVTec AD dataset;
2. PatchCore or the exact local patch-memory method used as the conceptual reference;
3. PaDiM;
4. industrial visual anomaly detection surveys or representative normal-only methods;
5. selective prediction and conditional computation;
6. any latency-aware or budgeted inference methods directly compared in the final manuscript.
