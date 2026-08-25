# Risk-Triggered Selective Anomaly Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 027 改造成一个面向 MVTec AD 的风险触发式选择性升级推理系统，并用固定延迟预算、异常召回率和 Full fallback 比例验证其有效性。

**Architecture:** 保留现有轻量 RATA 路径作为第一阶段，在 pooled Token 特征上增加正常性/异常风险评分；低风险样本直接输出，高风险样本调用 Full 模型复核。阈值由 validation 正常图像和预算曲线共同确定，测试集只做最终一次评估。CIFAR-100-C 保留为外部压力验证，不与 MVTec AD 的异常检测指标混合。

**Tech Stack:** PyTorch 2.8、CUDA 12.8、Python 3.12、NumPy、torchvision、JSON/YAML、unittest、RTX 3090。

---

## 文件结构与职责

- Create: `02_代码/mvtec_ad.py` — MVTec AD 目录验证、正常/异常样本读取和固定 validation 划分。
- Create: `02_代码/selective_inference.py` — anomaly score、阈值选择、fallback 调度和系统级结果计算。
- Create: `02_代码/evaluate_mvtec_selective.py` — MVTec AD 单类别/全类别评估入口。
- Create: `02_代码/tests/test_mvtec_ad.py` — 数据适配器测试。
- Create: `02_代码/tests/test_selective_inference.py` — 风险评分、阈值和 fallback 测试。
- Modify: `02_代码/rata_vit.py` — 暴露 pooled feature、cheap logits 和第一阶段 route metadata。
- Modify: `02_代码/real_cifar100.py` — 复用统一端到端延迟和 risk metadata 输出。
- Modify: `02_代码/evaluate_cifar100c_models.py` — 复用 selective evaluator 的风险统计字段。
- Create: `04_数据与划分/027_MVTec_AD数据清单_v1.md` — 数据版本、类别、划分和许可记录。
- Create: `04_分析与图表/027_selective_inference结果规范_v1.md` — 指标口径、预算曲线和统计报告规范。

### Task 1: 建立 MVTec AD 数据适配器

**Files:**
- Create: `02_代码/tests/test_mvtec_ad.py`
- Create: `02_代码/mvtec_ad.py`
- Create: `04_数据与划分/027_MVTec_AD数据清单_v1.md`

- [ ] **Step 1: Write the failing tests**

添加以下测试：

```python
def test_index_reads_good_train_and_test_anomaly_samples():
    dataset = MVTecADIndex(fake_root)
    self.assertEqual(dataset.categories(), ["bottle"])
    self.assertEqual(dataset.split_counts("bottle", "train"), {"good": 2})
    self.assertEqual(dataset.split_counts("bottle", "test"), {"good": 1, "broken": 1})

def test_validation_split_uses_only_good_training_images():
    train, validation = split_good_images(fake_train_paths, validation_fraction=0.5, seed=17)
    self.assertTrue(set(train).isdisjoint(validation))
    self.assertEqual(len(train) + len(validation), len(fake_train_paths))

def test_missing_category_structure_raises_clear_error():
    with self.assertRaises(FileNotFoundError):
        MVTecADIndex(missing_root).samples("bottle", "test")
```

- [ ] **Step 2: Run the tests and verify failure**

运行：

```powershell
python -m unittest 02_代码/tests/test_mvtec_ad.py -v
```

预期：由于 `mvtec_ad.py` 尚不存在而失败。

- [ ] **Step 3: Implement the minimal adapter**

实现 `MVTecADIndex(root)`，固定使用 `train/good`、`test/good`、`test/<defect>` 和 `ground_truth/<defect>`；实现 `categories()`、`samples(category, split)`、`split_counts()` 与 `split_good_images()`。样本记录至少包含 `image_path`、`category`、`split`、`is_anomaly` 和可选 `mask_path`。

- [ ] **Step 4: Validate against the uploaded dataset**

运行：

```powershell
python -m unittest 02_代码/tests/test_mvtec_ad.py -v
```

再在云端执行：

```bash
/root/miniconda3/bin/python /root/sj-tmp/rata_027/02_code/mvtec_ad.py --root /root/sj-tmp/rata_027/data/mvtec_ad --check
```

预期：15 个类别均能读取，输出正常训练样本数、正常测试样本数和异常测试样本数。

### Task 2: 暴露第一阶段风险特征

**Files:**
- Modify: `02_代码/rata_vit.py`
- Modify: `02_代码/tests/test_rata_vit.py`

- [ ] **Step 1: Add a failing metadata test**

要求 `RATAViT.forward()` 在不改变 logits 输出形状的情况下返回：

```python
route["pooled_feature"]
route["cheap_logits"]
route["difficulty"]
route["uncertainty"]
route["token_counts"]
```

测试 pooled feature 的形状为 `(batch, embed_dim)`，并确认 Full、Fixed、RATA 三种路径都能返回统一字段。

- [ ] **Step 2: Run the targeted test and verify failure**

运行：

```powershell
python -m unittest 02_代码/tests/test_rata_vit.py -v
```

- [ ] **Step 3: Implement metadata-only changes**

将已有 `pooled = patches.mean(dim=1)` 保存到 route；固定策略使用同维度的 pooled feature，不能在 Full/Fixed 路径重新计算异常模型。

- [ ] **Step 4: Run all existing tests**

运行：

```powershell
python -m unittest discover -s 02_代码/tests -v
```

预期：所有现有测试和新增测试通过。

### Task 3: 实现异常风险评分和阈值控制器

**Files:**
- Create: `02_代码/selective_inference.py`
- Create: `02_代码/tests/test_selective_inference.py`

- [ ] **Step 1: Write failing unit tests**

测试以下行为：

```python
def test_normal_prototype_score_is_zero_for_the_prototype():
    scorer = NormalityRiskScorer()
    scorer.fit(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    self.assertAlmostEqual(float(scorer.score(torch.tensor([[0.5, 0.5]])).item()), 0.0, places=5)

def test_threshold_selection_respects_fallback_budget():
    threshold = choose_threshold([0.1, 0.2, 0.8, 0.9], fallback_budget=0.5)
    self.assertGreaterEqual(threshold, 0.2)

def test_risk_fallback_calls_full_only_for_high_risk_samples():
    result = selective_predict(low_risk_scores, low_outputs, full_predict, threshold=0.5)
    self.assertEqual(result["fallback_count"], 2)
```

- [ ] **Step 2: Implement minimal components**

实现：

- `NormalityRiskScorer.fit(normal_features)`：保存均值和标准差或 prototype；
- `NormalityRiskScorer.score(features)`：输出标准化距离；
- `combine_risk(anomaly_score, uncertainty, weights)`：输出 `[0, 1]` 风险分数；
- `choose_threshold(validation_scores, fallback_budget)`：只基于 validation 分数确定阈值；
- `selective_predict()`：对低风险样本保留快速输出，对高风险样本调用 Full 函数并记录 fallback mask、总输出和两阶段延迟。

- [ ] **Step 3: Add calibration and latency accounting tests**

确认 `selective_predict()` 输出包含：`fallback_rate`、`corrected_count`、`new_error_count`、`mean_latency_ms`、`p95_latency_ms` 和每个样本的 `fallback` 标记。

- [ ] **Step 4: Run targeted and full tests**

运行：

```powershell
python -m unittest 02_代码/tests/test_selective_inference.py -v
python -m unittest discover -s 02_代码/tests -v
```

### Task 4: 接入 MVTec AD selective evaluator

**Files:**
- Create: `02_代码/evaluate_mvtec_selective.py`
- Modify: `02_代码/tests/test_mvtec_ad.py`

- [ ] **Step 1: Add a CPU one-batch integration test**

构造一个 `bottle` 的 fake index，使用 mock-free 的实际 `MVTecADIndex` 和当前模型，验证 evaluator 能输出：`image_auroc`、`recall_at_fpr`、`fallback_rate` 和 `p95_latency_ms`。

- [ ] **Step 2: Implement validation-only fitting**

训练阶段只读取 normal train images；从正常训练图像划出固定 validation 子集，拟合 normal prototype 和 risk threshold；不得读取 test labels。

- [ ] **Step 3: Implement four system modes**

实现 `full_only`、`sparse_only`、`random_fallback` 和 `risk_fallback`。`random_fallback` 使用固定 seed 产生与 Risk-fallback 相同的 fallback 数量，作为公平控制组。

- [ ] **Step 4: Run the three-category smoke**

云端运行：

```bash
/root/miniconda3/bin/python /root/sj-tmp/rata_027/02_code/evaluate_mvtec_selective.py \
  --data-root /root/sj-tmp/rata_027/data/mvtec_ad \
  --categories bottle cable hazelnut \
  --modes full_only sparse_only random_fallback risk_fallback \
  --device cuda \
  --output-root /root/sj-tmp/rata_027/runs/mvtec_selective_smoke
```

预期：每个类别四种模式均生成 JSON，且阈值来源标记为 validation。

### Task 5: 固定实验协议和统计输出

**Files:**
- Create: `04_分析与图表/027_selective_inference结果规范_v1.md`
- Modify: `02_代码/analysis_protocol.py`
- Create: `02_代码/tests/test_selective_statistics.py`

- [ ] **Step 1: Test paired bootstrap and budget alignment**

新增测试确认 bootstrap CI 可复现、Risk-fallback 与 Random-fallback 使用配对类别差值，并且阈值选择不读取 test labels。

- [ ] **Step 2: Implement statistics**

输出每个类别和 macro 平均的：AUROC、Recall、漏检率、fallback 率、平均延迟、P95 延迟及 95% CI。多类别比较使用类别配对 bootstrap，固定 seed=17 并记录 reps=10,000。

- [ ] **Step 3: Write the result specification**

固定主表、类别表、coverage-risk 曲线、延迟—召回 Pareto 图和失败案例表；禁止把 CIFAR Accuracy 与 MVTec AUROC 合成一个总分。

- [ ] **Step 4: Run all tests**

运行：

```powershell
python -m unittest discover -s 02_代码/tests -v
```

### Task 6: 云端三阶段实验

**Files:**
- Create: `/root/sj-tmp/rata_027/runs/mvtec_selective_smoke/`
- Create: `/root/sj-tmp/rata_027/runs/mvtec_selective_full/`
- Create: `/root/sj-tmp/rata_027/runs/mvtec_selective_summary.json`

- [ ] **Step 1: Run bottle/cable/hazelnut smoke**

只有四种模式、三类全部生成有效 JSON，才进入全类别实验。

- [ ] **Step 2: Run all 15 categories**

使用 3 个 seed，固定 validation 阈值流程；每个类别单独保存原始结果，不覆盖 smoke 目录。

- [ ] **Step 3: Run CIFAR-100-C severity 1/3/5**

使用已经固定的 risk threshold policy，不因 severity 结果重新调阈值；结果单独保存为外部压力验证。

- [ ] **Step 4: Aggregate and audit**

核对每个正文数字都来自 JSON；检查 fallback 比例、延迟预算和 CI；若成功判据未满足，论文结论自动收缩为风险—延迟权衡分析。

## 资源估计

- MVTec AD 数据：约 5,000 张高分辨率图像，先确认云端数据盘空间后再上传；
- 三类别 smoke：约 15–30 分钟；
- 全 15 类、4 种系统、3 seed：预计 2–6 GPU 小时，按类别并行但限制 CPU worker；
- CIFAR-100-C severity 1/3/5：已有数据和代码，预计 30–60 分钟；
- 不下载 MVTec AD 2，直到 MVTec AD 主实验通过。

## 验收标准

实现完成必须同时满足：

1. 本地和云端单元测试全部通过；
2. bottle/cable/hazelnut smoke 四种模式全部可复现；
3. Risk-fallback 的阈值只来自 validation；
4. 每个结果包含 fallback 比例和端到端 P95；
5. 全类别结果包含 bootstrap 95% CI；
6. 论文主张根据预注册成功判据自动收缩或放宽，不手工挑选类别。

