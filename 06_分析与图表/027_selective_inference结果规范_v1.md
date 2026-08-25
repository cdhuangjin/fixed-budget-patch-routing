# 027 风险触发式选择性推理结果规范 v1

## 主问题

在同一端到端延迟预算下，轻量第一阶段加风险触发 Full fallback 是否能降低工业异常检测漏检率，并保持可解释的升级比例。

## 数据与校准

- MVTec AD 的 `train/good` 只用于拟合正常性特征，并固定划分 validation。
- 阈值、fallback 预算和延迟预算只能由 validation 选择；测试集标签不得参与调参。
- MVTec AD 与 CIFAR-100-C 分开报告，不合并 AUROC 与分类 Accuracy。

## 必报指标

每个类别、macro 平均及 95% bootstrap CI（seed=17，reps=10,000）报告：image AUROC、FPR=5% 时召回率、漏检率、fallback rate、平均端到端延迟、P95 端到端延迟。

## 对照组

`full_only`、`sparse_only`、与 Risk-fallback 具有相同 fallback 数量的 `random_fallback`、`risk_fallback`。随机组使用固定 seed=17。

## 主表与图

主表报告四种系统的 AUROC、Recall@5% FPR、漏检率、fallback rate、mean/P95 latency；类别表保留每一类原始结果；绘制 coverage-risk 曲线、延迟—召回 Pareto 图和失败案例表。

## 结论约束

只有在统一 P95 延迟预算下 Risk-fallback 在 macro 召回或漏检率上优于 random_fallback，并且 bootstrap CI 不跨越零，才使用“风险触发升级改善异常检测”的强主张。否则收缩为风险—延迟权衡和类别依赖性分析。
