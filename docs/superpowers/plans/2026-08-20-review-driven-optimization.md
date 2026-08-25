# 027 Review-Driven Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 027 的工业异常检测论文从“Risk 优于 Random”扩展为可审计的风险分配研究，并消除成本定义、基线、路由消融、定位和泛化证据上的主要审稿风险。

**Architecture:** 保留现有 Fast/Full/Risk/Random 主协议；新增统一的路由策略接口、matched-budget 消融、真实端到端延迟汇总和区域级定位评估。论文只把 Risk > matched Random 作为主因果结论，Full 作为质量上限，不再把 Risk 描述成逐样本加速器。

**Tech Stack:** Python 3.12、PyTorch 2.8、torchvision、NumPy、PIL、unittest、JSON、MVTec AD。

---

### Task 1: 建立可追溯的实验计划与输出结构

**Files:**
- Create: `00_方案/027_审稿意见驱动优化实验矩阵_v1.md`
- Create: `05_运行记录/README_结果目录规范_v1.md`

- [ ] **Step 1: 固定主问题、终点和比较对象**

记录唯一主问题：在相同实际回退率下，风险选择是否优于随机选择。固定主终点为 image-level AUROC，次终点为 Recall@5%FPR、实际回退率、端到端 P50/P95 和 pixel/PRO 指标。

- [ ] **Step 2: 固定单变量消融矩阵**

使用同一 Fast/Full、同一数据划分、同一 seed，只改变路由信号：`risk_fast_score`、`uncertainty_proxy`、`random_matched`、`oracle_test_label_upper_bound`。Oracle 只作上界，不进入主结论。

- [ ] **Step 3: 固定资源估计和失败判据**

先运行本地单元测试和静态检查；MVTec 上传后运行 3 类 smoke，再运行 15 类 × 3 seed 的路由消融、PRO 和统一 latency。若外部数据仍不可用，论文明确降级为单数据集应用研究。

### Task 2: 实现统一路由策略与匹配预算控制

**Files:**
- Create: `02_代码/selective_routes.py`
- Create: `02_代码/tests/test_selective_routes.py`
- Modify: `02_代码/evaluate_mvtec_patchcore.py`

- [ ] **Step 1: Write tests for deterministic routing**

测试固定阈值、匹配回退数量、随机种子复现、以及任何策略都不读取测试标签。

- [ ] **Step 2: Implement route interface**

实现 `RouteDecision`、`risk_route(scores, threshold)`、`random_matched_route(n, fallback_count, seed)`、`uncertainty_route(scores, fallback_count)` 和 `oracle_route(labels, fallback_count)`；所有路由返回 `mask`、`threshold`、`target_count`、`actual_rate`。

- [ ] **Step 3: Refactor evaluator to use the interface**

保留当前 JSON 字段兼容性；新增 `route_name`、`target_fallback_count`、`actual_fallback_count` 和 `route_source`，并将 Random 改为按精确数量匹配，而不是 Bernoulli 采样后才比较比例。

### Task 3: 增加统一的端到端延迟和 memory-bank 审计

**Files:**
- Create: `02_代码/benchmark_mvtec_unified.py`
- Create: `02_代码/tests/test_latency_summary.py`
- Modify: `02_代码/benchmark_mvtec_selective_latency.py`

- [ ] **Step 1: Define one latency protocol**

固定 batch=1、20 warmup、至少 100 次有效重复、CUDA synchronize、正式 memory bank 规模；分别报告 Fast、Full、Risk、Random 的 P50、P95、mean、IQR、fallback rate 和 memory-bank size。

- [ ] **Step 2: Implement summary output**

输出 `latency_protocol`、`bank_images_by_category`、`n_images`、每种 route 的完整统计，并明确 Risk 是顺序 Fast→Full 的端到端时间。

- [ ] **Step 3: Add CPU-safe tests**

测试 percentile 汇总、空样本报错、字段完整性，不在单元测试中依赖 CUDA。

### Task 4: 增加像素级 PRO/区域级评估

**Files:**
- Create: `02_代码/evaluate_mvtec_pro.py`
- Create: `02_代码/tests/test_mvtec_pro.py`

- [ ] **Step 1: Implement connected-component overlap**

对每个异常 mask 的连通区域计算 PRO 曲线，按固定 FPR 网格积分；只使用测试 mask 计算评价，不参与阈值校准。

- [ ] **Step 2: Evaluate Full and Risk-triggered localization**

Full 对全部测试图像计算定位图；Risk 对未回退样本记录 `localization_available=false`，不得把 Full 定位结果冒充 Risk 定位结果。

- [ ] **Step 3: Test numerical edge cases**

测试无异常区域、全零预测、单像素区域和空预测，确保输出有限值或明确 NaN 状态。

### Task 5: 生成审稿级统计与图表数据

**Files:**
- Create: `02_代码/summarize_mvtec_review.py`
- Create: `02_代码/tests/test_summarize_mvtec_review.py`
- Modify: `02_代码/analysis_protocol.py`

- [ ] **Step 1: Implement paired and clustered summaries**

按类别—种子保存原始差值，报告 macro mean、类别配对 bootstrap CI、seed-level mean 和正向类别数；不把 45 个单元误写成 45 个独立数据集。

- [ ] **Step 2: Implement required plots data**

输出 budget-quality、risk-vs-random per-category、latency-quality 和 failure-case CSV/JSON 数据。

- [ ] **Step 3: Verify report fields**

缺少外部数据、PRO 或端到端延迟时，脚本应显式输出 `AUTHOR_INPUT_NEEDED`，不能静默生成完整性结论。

### Task 6: 修订论文和正式报告

**Files:**
- Modify: `07_论文/027_论文草稿_v2_风险触发选择性工业异常检测.md`
- Modify: `06_实验报告/027_风险触发选择性推理正式报告_v2.md`
- Create: `07_论文/027_claim_evidence_matrix_v3.md`

- [ ] **Step 1: 统一成本定义**

摘要和主表只使用端到端 latency 或明确标注 estimated path cost；保留 Risk P95 高于 Full 的结果。

- [ ] **Step 2: 收窄主张**

主张限定为风险分配优于匹配 Random，不宣称 Risk 超过 Full、降低逐样本 P95 或跨工业场景泛化。

- [ ] **Step 3: 增加证据矩阵**

每个主要 claim 对应原始 JSON、统计脚本、指标定义和当前状态；缺失项标记为 `unsupported` 或 `AUTHOR_INPUT_NEEDED`。

### Task 7: 验证与发布前审计

**Files:**
- No new files; verify all modified files.

- [ ] **Step 1: Run all unit tests**

运行 `python -m unittest discover -s 02_代码/tests -v`。

- [ ] **Step 2: Compile all 027 Python files**

运行 `python -m compileall -q 02_代码`。

- [ ] **Step 3: Audit claim consistency**

检查论文、正式报告、claim matrix 中的数字、延迟口径、数据集范围和限制是否一致。

- [ ] **Step 4: Report unresolved external-data gaps**

没有 MVTec 外部上传或外部工业数据时，不将代码完成误报为实验完成。
