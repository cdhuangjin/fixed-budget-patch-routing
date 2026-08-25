# 027 风险触发式选择性异常推理设计

## 1. 研究定位

将 027 从“单一动态稀疏模型在固定延迟下超过 Full”调整为“风险触发式选择性升级推理系统”。系统先使用轻量稀疏路径进行快速判断，仅对高风险样本调用 Full/高预算路径，从而在工业异常检测场景中控制平均计算成本和异常漏检风险。

英文暂定题目：**Risk-Triggered Selective Inference for Latency-Constrained Industrial Anomaly Detection**。

核心研究问题：

> 在相同平均延迟或 P95 延迟约束下，风险触发式选择性升级是否能够以较低 Full fallback 比例，降低工业异常样本的漏检率？

## 2. 系统结构

```text
输入图像
    ↓
轻量 Token 路径
    ↓
异常风险评分
    ├── 低风险：快速输出
    └── 高风险：Full / 高预算路径复核
```

### 模块 A：异常风险评分头

- 复用现有 cheap router 和 pooled Token 特征；
- 使用正常训练图像建立 normal prototype 或正常特征统计量；
- 输出 anomaly score，并与预测熵/置信度合成为风险分数；
- 不使用测试集标签调整风险分数。

### 模块 B：风险触发 Full fallback

- 第一阶段由轻量稀疏模型完成快速推理；
- 当风险分数超过 validation 集确定的阈值时，调用 Full 模型复核；
- 记录每张图像是否 fallback、第一阶段和第二阶段输出、总延迟；
- 低风险样本不重复计算。

### 模块 C：延迟预算控制器

- 只使用 validation 集选择风险阈值；
- 目标约束包括平均延迟预算和 P95 延迟预算；
- 在满足预算的候选阈值中最大化异常召回率或最小化漏检率；
- 测试集只用于一次最终报告，不用于阈值调参。

## 3. 数据与任务

### 主应用数据

使用完整 MVTec AD。训练阶段只使用每个类别的正常图像，测试阶段使用正常和缺陷图像。第一版只做 image-level anomaly detection，暂不加入像素级定位，以控制迁移范围。

### 机制压力测试

保留现有 CIFAR-100-C severity 1、3、5，用于验证风险触发机制在分布扰动下的行为，但不把 CIFAR-100-C 作为工业应用结论的替代品。

### 固定划分

- MVTec AD 原始 train/test 结构保留；
- 从正常训练图像中划分 validation 子集用于阈值和 normal prototype 选择；
- 测试异常标签、类别标签和缺陷类型不得参与阈值选择；
- 每个类别单独报告，同时报告 15 类 macro 平均。

## 4. 对比系统

| 系统 | 第一阶段 | 高风险处理 | 目的 |
|---|---|---|---|
| Full-only | Full | 无 | 精度和延迟参考上界 |
| Sparse-only | 轻量稀疏 | 无 | 低成本基线 |
| Random-fallback | 轻量稀疏 | 随机触发 Full | 排除随机升级收益 |
| Risk-fallback | 轻量稀疏 | 风险触发 Full | 本文方法 |
| Oracle-fallback | 轻量稀疏 | 使用测试标签触发 | 仅作为不可部署上界，不进入主比较 |

## 5. 主要指标

主指标：

- Image-level AUROC；
- 固定 FPR 下的异常 Recall；
- 异常漏检率；
- 平均端到端延迟；
- P95 端到端延迟；
- Full fallback 比例。

次指标：

- 每类别 AUROC 和 Recall；
- 正常样本误报率；
- token 保留率；
- 峰值显存；
- 风险分数校准误差；
- 第一阶段错误、fallback 后纠正率和 fallback 后新增错误率。

## 6. 预注册成功判据

只有同时满足以下条件，才能使用“风险触发式选择性升级有效”的表述：

1. Risk-fallback 在相同平均延迟或相同 P95 预算下，异常 Recall 不低于 Sparse-only；
2. Risk-fallback 的异常漏检率低于 Random-fallback；
3. fallback 比例低于 50%，并在至少 10/15 个 MVTec AD 类别上方向一致；
4. 主要差异的 bootstrap 95% CI 不跨 0，或在类别配对检验中达到预设显著性标准；
5. CIFAR-100-C 至少两个 severity 条件下保持同方向。

如果无法满足上述判据，论文主张收缩为“工业异常检测中的选择性升级行为与延迟—风险权衡分析”。

## 7. 实验阶段

### 阶段 0：数据和接口 smoke

- 验证 MVTec AD 目录结构；
- CPU 读取一个类别的正常/异常图像；
- 完成一个 batch 的风险评分、fallback 和指标计算。

### 阶段 1：单类别机制验证

- 选择 bottle、cable、hazelnut 三类；
- 比较四种系统；
- 固定平均延迟预算和 P95 预算各跑一组；
- 确认 fallback 逻辑有效后再扩展全部类别。

### 阶段 2：全 MVTec AD

- 15 类完整测试；
- 3 个随机 seed；
- 输出类别级表格、延迟—召回曲线和风险分数分布图。

### 阶段 3：CIFAR-100-C 外部压力验证

- severity 1、3、5；
- 只使用已经固定的风险阈值策略；
- 不因单一 severity 结果修改系统结构。

## 8. 边界与风险

- 第一版不声称像素级异常定位；
- 不把 MVTec AD 的无监督异常检测结果与 CIFAR 分类 Accuracy 直接合并；
- 不使用测试标签选择阈值或 fallback 样本；
- 不把 Full fallback 比例降低本身当作性能提升；
- 如果风险评分不能识别真正高风险样本，优先报告失败模式，不继续堆叠模块。

