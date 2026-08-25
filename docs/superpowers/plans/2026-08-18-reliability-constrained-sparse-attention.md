# Reliability-Constrained Sparse Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 027 改造成“固定真实 P95 延迟预算下提升困难样本和长尾类别性能”的可验证稀疏注意力实验管线。

**Architecture:** 将 patch embedding、token routing、稀疏 self-attention、分类头和评估指标拆成清晰接口。Full 使用全 token attention，Fixed/Random 使用固定 token 子序列，Difficulty-only、Uncertainty-only 和 RATA 使用不同路由信号；所有路径都用同一延迟测量器记录端到端 P50/P95。

**Tech Stack:** Python 3.12, PyTorch 2.8, CUDA 12.8, torchvision CIFAR-100/CIFAR-100-C, unittest, RTX 5090 D.

---

### Task 1: 建立稀疏注意力模块接口

**Files:**
- Create: `02_代码/sparse_attention.py`
- Modify: `02_代码/tests/test_rata_vit.py`

- [ ] **Step 1: Write the failing tests**

新增测试覆盖：全 token 输出形状、固定 token 子序列输出形状、不同 token 数均可反向传播，以及 token 数确验影响 attention 输入长度。

- [ ] **Step 2: Run the focused tests and verify failure**

运行：`python -m unittest 02_代码.tests.test_rata_vit -v`

预期：因 `sparse_attention.py` 不存在而失败。

- [ ] **Step 3: Implement the minimal attention module**

实现 `SparseSelfAttention(embed_dim, heads)`，接受形状 `[B, N, D]` 的序列，使用 `torch.nn.functional.scaled_dot_product_attention` 计算 Q/K/V，并返回同形状输出；实现 `SparseTransformerBlock`，由 attention、LayerNorm、MLP 和残差组成。

- [ ] **Step 4: Run focused tests**

运行同一测试命令，预期新增测试全部通过。

### Task 2: 重构 RATA 模型的 Full/Fixed/Adaptive 路径

**Files:**
- Modify: `02_代码/rata_vit.py`
- Modify: `02_代码/smoke_train.py`
- Modify: `02_代码/tests/test_rata_vit.py`

- [ ] **Step 1: Add route-policy tests**

测试 `full` 返回全部 patch、`fixed_sparse` 返回固定前 k 个 patch、`random_sparse` 在固定种子下可复现、`difficulty` 和 `uncertainty` 分别只调用对应信号、`rata` 同时使用两个信号。

- [ ] **Step 2: Run tests and verify failure**

预期：现有 `build_model` 不支持六种方法，测试失败。

- [ ] **Step 3: Implement explicit route policy**

增加 `route_policy` 配置字段和 `select_tokens(patches, policy)` 接口；所有方法均进入 `SparseTransformerBlock`，不得调用旧的 `nn.TransformerEncoder`。固定策略不计算 router，RATA 才计算困难度、不确定性和 token score。

- [ ] **Step 4: Run all model tests**

运行：`python -m unittest discover -s 02_代码/tests -v`

预期：所有模型、形状、梯度和路由测试通过。

### Task 3: 扩展六组训练与评估接口

**Files:**
- Modify: `02_代码/real_cifar100.py`
- Modify: `02_代码/tests/test_real_cifar100.py`

- [ ] **Step 1: Write failing metric tests**

用人工 logits 和标签测试：困难样本 CE 排序、最差 10% 准确率、按类别 Macro-F1、ECE、每类样本数和长尾分位数统计。

- [ ] **Step 2: Run tests and verify failure**

预期：缺少 `macro_f1`、`ece` 和困难样本分组字段而失败。

- [ ] **Step 3: Implement deterministic metrics**

新增 `compute_class_metrics`、`compute_ece` 和 `summarize_hard_examples`；结果 JSON 必须记录 method、seed、token_keep_ratio、accuracy、worst_10pct_accuracy、macro_f1、ece、latency 和类别分布。

- [ ] **Step 4: Add six-method CLI validation**

CLI 方法限制为 `full,fixed_sparse,random_sparse,difficulty_only,uncertainty_only,rata`；训练前固定 seed，评估不使用测试标签调阈值。

- [ ] **Step 5: Run all tests**

运行：`python -m unittest discover -s 02_代码/tests -v`

预期：测试全部通过。

### Task 4: 增加 CIFAR-100-C 评估和固定延迟配对

**Files:**
- Create: `02_代码/evaluate_cifar100c.py`
- Create: `02_代码/tests/test_evaluate_cifar100c.py`
- Modify: `03_配置/027_main.yaml`

- [ ] **Step 1: Write failing loader and severity tests**

测试能读取 severity 1、3、5，输出 `[method, severity, corruption, metric]` 结构，并拒绝使用测试集标签调节 token 阈值。

- [ ] **Step 2: Implement checkpoint evaluation**

加载各方法 checkpoint，在相同 batch、相同 GPU 上测量端到端延迟；按照 Full 的 P95 延迟预算筛选可比运行点，再计算总体准确率、困难样本准确率、Macro-F1 和 ECE。

- [ ] **Step 3: Update configuration**

把六种方法、三种 seed、三种 severity、固定 P95 配对规则写入 `027_main.yaml`。

- [ ] **Step 4: Run loader and smoke tests**

先用已有 CIFAR-10-C 目录做路径检查；缺少 CIFAR-100-C 时只报告阻塞，不把 CIFAR-10-C 冒充 CIFAR-100-C。

### Task 5: 5090 延迟 smoke 和小规模训练矩阵

**Files:**
- Modify: `02_代码/benchmark_latency.py`
- Create: `02_代码/run_stage1_matrix.py`
- Create: `05_运行记录/stage1_manifest.json`

- [ ] **Step 1: Add latency tests**

验证每种方法均记录 warmup、repeats、GPU 名称、batch size、P50、P95 和 token keep ratio。

- [ ] **Step 2: Run 5090 latency smoke**

固定 batch=32、224×224、warmup=10、repeats=50，先完成六种方法；若稀疏方法的 P95 没有低于 Full，则仍可继续困难样本实验，但不得宣称延迟收益。

- [ ] **Step 3: Implement stage-1 matrix runner**

按 `seed ∈ {5,17,29}`、六种方法顺序运行，每个运行写入 `/root/autodl-tmp/rata_027/05_运行记录/stage1/<seed>/<method>/`，失败只写失败 JSON 并继续后续独立运行。

- [ ] **Step 4: Launch cloud matrix**

使用 `/root/miniconda3/bin/python` 和 `/root/autodl-tmp/rata_027/data`；不得读取或写入 `/root/autodl-tmp/shift_cp_021`。

### Task 6: 分析、验证和投稿判定

**Files:**
- Create: `05_运行记录/stage1_summary.json`
- Create: `04_分析/stage1_analysis.md`

- [ ] **Step 1: Aggregate three-seed results**

计算均值、标准差和 bootstrap 95% CI，分别比较 Full、Fixed、Random 和 RATA 的总体准确率、困难样本准确率、Macro-F1、ECE 与 P95。

- [ ] **Step 2: Check success criteria**

只有在固定 Full P95 预算下，RATA 困难样本准确率和长尾 Macro-F1 稳定优于两个稀疏控制组，且总体准确率下降不超过 1 个百分点时，才进入 ImageNet-100。

- [ ] **Step 3: Preserve negative results**

若条件不满足，保留所有原始 JSON 和 checkpoint，并将结论写成“当前路由策略未形成稳定收益”，不继续扩大昂贵实验。

