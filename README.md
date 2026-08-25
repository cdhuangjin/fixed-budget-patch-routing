# Experiment 027: Fixed-Budget Local-Patch Risk Routing for Industrial Anomaly Detection

This repository contains the source code, analysis materials, and manuscript sources for Experiment 027. The study evaluates whether a local-patch risk probe can allocate a fixed fallback budget for industrial anomaly detection.

## 项目定位

研究输入难度和不确定性是否可以共同决定注意力/token 计算量，使模型在相同真实 GPU 延迟预算下获得更高准确率，并减少难例性能退化。

英文题目：**Adaptive Sparse Attention and the Accuracy–Efficiency Frontier**

投稿方向：高效深度学习、视觉 Transformer、模型压缩与可靠推理。

## 核心主张边界

本文不把理论 FLOPs 或平均吞吐当作唯一效率证据。核心比较必须在同一 GPU、同一 batch 和相同真实 P50/P95 延迟预算下进行，并同时报告难例准确率、token 保留率、峰值显存和能耗代理。

## 当前状态

- 阶段：clean CIFAR-100 正式基线已完成；当前转入风险感知动态 Token 推理的工业视觉应用验证。
- 硬件：本地 NVIDIA GeForce RTX 5060 (8GB)。
- 首轮数据：clean CIFAR-100 机制验证；工业视觉应用数据待接入。CIFAR-100-C 大文件下载已暂停，不作为当前论文依赖。
- 正式随机种子：5、17、29、41、53。
- 主终点：固定真实 GPU 延迟预算下的准确率。

## 目录

- `00_方案/`：正式协议、实验矩阵、资源预算和运行检查。
- `01_文献/`：稀疏注意力、动态 token、难度估计和真实延迟文献核验。
- `02_代码/`：模型、门控、训练、延迟基准和评估脚本。
- `03_配置/`：模型、稀疏率、数据和 seed 配置。
- `04_数据与划分/`：数据清单、固定划分和难度分层。
- `05_运行记录/`：训练日志、GPU 资源、延迟和 checkpoint。
- `06_分析与图表/`：准确率—延迟 Pareto 曲线、难例分析和消融图。
- `07_论文/`：论文草稿、补充材料和投稿版本。

## 第一阶段停止条件

若自适应策略在等真实 P95 延迟下不能提高准确率，或只在平均样本有效而在最坏 10% 难例上退化，则不扩展到 ImageNet-100，优先完成工业应用数据上的困难/异常样本验证，并收缩论文主张。

## 权威主表来源（统一口径）

- 统计闭环报告：`06_主线项目/027_自适应稀疏注意力与准确率效率前沿/05_运行记录/stats_report.json`

为避免不同报告之间出现数字冲突，后续论文、审计和汇报均以以下文件为主表权威来源：

- `05_运行记录/canonical_main_table.json`（MVTec多seed + VisA seed5 + MPDD seed5 的选择性推理汇总）
- `06_分析与图表/top_tier_metrics.json`（待补充）

说明：
- 当前 canonical 主表覆盖 MVTec 45 条件、VisA 12 条件、MPDD 6 条件；
- 主终点为同 fallback budget 下的 risk/random image AUROC、risk delta、fallback rate 与 risk≥fast 比例；
- 若后续新增更多 seed 或新增 strict_quota 主表，应先更新 canonical 文件再同步主结论。
## 统计口径与当前主结论

- 所有主结论均以 05_运行记录/stats_report.json 中的 dataset 聚合统计为准；
- 当前可用于主表的结论应区分“强结论”与“边界结论”，避免过度推广。
