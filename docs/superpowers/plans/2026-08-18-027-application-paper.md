# 027 Application Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 027 从“超过 Full 的稀疏注意力算法”收敛为“风险感知动态 Token 推理”的应用型论文，并用可复现的 clean CIFAR-100 机制实验和一个真实视觉应用数据集支撑论文主张。

**Architecture:** 保留现有 RATA-ViT、Full、Fixed Sparse、Random Sparse、Difficulty-only 和 Uncertainty-only 六种策略。论文分两层证据：第一层用 clean CIFAR-100 验证路由机制和稳定性；第二层用工业视觉数据验证困难样本、长尾/异常样本与真实延迟约束下的行为。CIFAR-100-C 暂停，不作为当前论文依赖。

**Tech Stack:** PyTorch 2.8, CUDA 12.8, Python 3.12, RTX 5090 D, NumPy, JSON/YAML, pytest, Markdown/LaTeX。

---

### Task 1: 固化 clean CIFAR-100 证据边界

**Files:**
- Create: `04_分析与图表/正式基线结果_v1.md`
- Modify: `README.md`

- [ ] **Step 1: 记录当前可支持的主张**

写入原 Stage 1 三种子与新增正式两种子的结果范围，明确 RATA 只相对稀疏控制组有优势，不相对 Full 有优势；100-step smoke 结果不得进入正式统计。

- [ ] **Step 2: 标记不可支持的主张**

明确禁止以下表述：固定 P95 下超过 Full、困难样本显著提升、跨数据集泛化、真实工业部署收益。

- [ ] **Step 3: 更新项目状态**

将 README 的“尚未启动正式训练”改为“clean CIFAR-100 正式基线已完成，工业视觉应用验证待执行；CIFAR-100-C 下载暂停”。

- [ ] **Step 4: 检查文档一致性**

确认 README、正式实验协议和结果分析中的方法名称、六个 baseline、seed 列表和主终点一致。

### Task 2: 建立应用型论文实验矩阵

**Files:**
- Create: `07_论文/027_论文路线与实验矩阵_v1.md`
- Modify: `00_方案/027_正式实验协议_v2.md`

- [ ] **Step 1: 固定应用论文问题**

论文问题固定为：在边缘视觉推理的延迟预算下，风险感知的动态 Token 分配是否能把更多计算分配给困难/异常样本，并降低简单样本的平均计算开销。

- [ ] **Step 2: 固定实验层级**

第一层：clean CIFAR-100 六方法、5 个正式 seed、平均准确率/Macro-F1/P50/P95/ECE。

第二层：一个工业视觉数据集，至少包含正常/异常或多类别缺陷；报告异常样本召回率、困难样本召回率、Macro-F1、P95 延迟和 token 保留率。

第三层：只对最终赢家做一次跨数据集或输入压力验证，不在验证集之外调阈值。

- [ ] **Step 3: 固定消融顺序**

按 `Full → Fixed Sparse → Random Sparse → Difficulty-only → Uncertainty-only → RATA` 顺序执行；应用数据只保留 Full、Fixed Sparse、RATA 和去掉一个门控分支的两项核心消融，避免实验矩阵失控。

- [ ] **Step 4: 固定成功判据**

只有在相同 P95 延迟预算下，RATA 同时不低于 Fixed Sparse 且困难/异常样本指标改善，并且至少两个 seed 或两个数据子集方向一致时，才能使用“风险感知收益”表述；否则写成“路由行为分析”。

### Task 3: 准备论文骨架和证据映射

**Files:**
- Create: `07_论文/027_论文初稿骨架_v1.md`

- [ ] **Step 1: 写四段式引言骨架**

按“边缘视觉延迟约束 → 简单样本与困难样本计算需求不均 → 现有稀疏方法的实际延迟/风险缺口 → RATA 与工业应用验证”组织，不把 Full 超越写成前提。

- [ ] **Step 2: 建立方法与代码映射**

将 `rata_vit.py`、`sparse_attention.py`、`real_cifar100.py` 和 `benchmark_latency.py` 分别映射到路由、稀疏注意力、训练评估和延迟测量小节。

- [ ] **Step 3: 建立结果表占位结构**

表 1 为 clean CIFAR-100 主结果，表 2 为工业视觉应用结果，表 3 为门控消融，图 1 为路由流程，图 2 为准确率—P95 延迟关系，图 3 为困难/异常样本的 token 分配。

### Task 4: 选择并接入应用数据集

**Files:**
- Create: `04_数据与划分/027_应用数据集选择记录_v1.md`
- Modify: `02_代码/real_cifar100.py` only after the dataset interface is fixed
- Test: `02_代码/tests/test_application_dataset_adapter.py`

- [ ] **Step 1: 先做数据集可得性检查**

优先检查 MVTec AD 和 VisA 的现有云端缓存、官方可下载地址及磁盘占用；若完整数据不可用，先选能在 50 GB 数据盘内完成下载和解压的数据集，不再次启动未验证的大文件下载。

- [ ] **Step 2: 固定数据划分**

记录类别、训练/验证/测试划分、异常比例、图像分辨率和预处理；阈值和路由超参数只能由验证集选择。

- [ ] **Step 3: 写适配器测试**

测试样本、标签、异常标记和类别名称能被统一转换为当前评估接口，并能在 CPU 上完成一个 batch 的读取。

- [ ] **Step 4: 再启动小规模云端 smoke**

只使用 20–50 个 batch 验证路径、标签和输出指标，确认无误后才启动正式训练。

### Task 5: 生成可投稿图表和统计

**Files:**
- Create: `06_分析与图表/027_主结果表生成说明_v1.md`
- Create: `06_分析与图表/plot_clean_baseline.py` after result aggregation interface is fixed
- Test: `06_分析与图表/tests/test_plot_clean_baseline.py`

- [ ] **Step 1: 固定统计口径**

准确率、Macro-F1、ECE、P50/P95 延迟均报告 mean ± standard deviation；延迟必须来自相同 batch、warmup 和 repeat 配置。

- [ ] **Step 2: 固定图表**

生成方法—准确率、方法—P95 延迟和 accuracy–latency Pareto 图；所有图表从 JSON 结果读取，不手填数值。

- [ ] **Step 3: 做论文前审计**

检查每一个正文数字都能回溯到结果 JSON，每个结论都标注数据集、seed 和评估脚本。

---

## 资源估计

- clean CIFAR-100 已完成，新增正式种子实验约 8–10 分钟/批次，已使用独立输出目录。
- 应用数据 smoke 预计少于 10 分钟；正式应用实验按单 GPU 小规模矩阵控制在 2–6 GPU 小时。
- 不再下载 17.5 GB CIFAR-100-C 全量镜像；保持 027 数据目录和 021 数据目录完全分离。

## 分析计划

主比较：RATA vs Fixed Sparse；参考上界：Full；机制对照：Difficulty-only、Uncertainty-only、Random Sparse。

主指标：相同 P95 预算下的整体准确率和困难/异常样本 Macro-F1。次指标：P50/P95、ECE、token 保留率、峰值显存和类别/异常分层召回率。若应用数据结果不支持风险感知收益，论文主张收缩为“动态路由行为与效率权衡的验证分析”。
