# 027 Reliability-Constrained Sparse Attention 设计

## 1. 研究问题

在相同真实 GPU P95 延迟预算下，可靠性约束的 token 选择是否能够提升困难样本、长尾类别和分布偏移样本的识别性能？

论文主张不再设定为“RATA 必然比 Full Attention 更快或总体准确率更高”，而是比较固定延迟预算下的困难样本收益。

## 2. 方法设计

RATA 保留两个路由信号：样本困难度与预测不确定性。路由器输出 token 重要性分数，并为每个样本分配 token 预算。被选 token 进入真正的稀疏注意力计算路径；Full、Fixed Sparse 和 RATA 使用可区分的计算实现，避免所有方法都支付完整路由和完整注意力的隐性开销。

实现优先采用 PyTorch scaled dot-product attention 的 token 子序列计算。第一阶段只支持单 GPU、固定 batch、无动态 batch 合并，以保证延迟测量可复现。所有延迟均包含路由、索引、注意力和输出头，不用 FLOPs 替代真实延迟。

## 3. 实验对照

| 方法 | 路由策略 | 目的 |
|---|---|---|
| Full Attention | 全部 token | 精度上限 |
| Fixed Sparse | 固定 token 数 | 固定稀疏控制组 |
| Random Sparse | 随机 token | 随机选择控制组 |
| Difficulty-only | 仅困难度 | 单信号消融 |
| Uncertainty-only | 仅不确定性 | 单信号消融 |
| RATA | 困难度 + 不确定性 | 完整方法 |

## 4. 数据和指标

第一阶段使用 CIFAR-100，3 个随机种子；CIFAR-100-C 使用严重度 1、3、5。只有第一阶段出现稳定优势后，才升级 ImageNet-100。

主指标：

1. 固定 P95 延迟下的总体准确率；
2. 固定 P95 延迟下最差 10% 困难样本准确率；
3. 长尾类别 Macro-F1；
4. CIFAR-100-C 损坏鲁棒性。

辅助指标包括 ECE、token 保留率、吞吐、P50/P95 延迟和显存峰值。

## 5. 成功标准

在与 Full 相同的 P95 延迟预算下，RATA 的困难样本准确率和长尾 Macro-F1 稳定高于 Fixed Sparse 与 Random Sparse；总体准确率相对 Full 的下降不超过 1 个百分点。若不满足，则不把“效率—困难样本收益”作为主要结论。

## 6. 实验矩阵和资源

第一阶段为 18 组：6 种方法 × 3 个随机种子；CIFAR-100-C 在完成训练后复用 checkpoint，评估 3 个严重度，不额外重复训练。每组先运行 5,000 步短实验，预计单卡约 1–2 小时总墙钟时间，结果全部写入 027 的 `05_运行记录/`。

## 7. 验证和失败处理

- 先用单 batch forward 验证 Full、Fixed Sparse、RATA 的 token 数、logits 形状和梯度有限性。
- 再做 5090 D 上的 50 次延迟 smoke，确认测量包含路由和稀疏注意力。
- 再启动 3 个种子的训练矩阵。
- 任一方法失败时只修复该方法的隔离输出目录，不覆盖已有结果。
- 结果分析前保留原始 JSON、checkpoint 和硬样本索引。
