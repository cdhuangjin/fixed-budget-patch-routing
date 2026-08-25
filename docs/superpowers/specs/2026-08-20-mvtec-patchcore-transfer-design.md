# MVTec AD 强基线轻量改造与场景迁移实验设计

## 研究问题

在冻结的 ImageNet 预训练 WideResNet50-2 + PatchCore 强基线上，加入少量针对工业视觉场景的模块，能否在 MVTec AD 的图像级和像素级异常检测中提升鲁棒性，同时保持推理开销可控？

## 基线与模块

- Base：WideResNet50-2 的 layer2/layer3 patch 特征，正常训练图像建立 memory bank，最近邻距离作为异常分数。
- M1 多尺度特征融合：对 layer2 与 layer3 分别归一化、空间对齐后拼接，避免只依赖单层纹理或语义。
- M2 邻域一致性评分：不改变 memory bank，仅将单一最近邻距离替换为 3 个最近邻距离的均值，降低偶然近邻噪声。
- M3 光照/域偏移一致性：对正常训练图像施加轻微亮度与对比度扰动，要求原图和扰动图的 patch 表征一致；测试时对原图和轻微扰动得分做稳健聚合。

## 单变量矩阵

| Run | 变化 | 固定项 | 目的 |
|---|---|---|---|
| Base | 无模块 | 数据划分、backbone、memory budget、seed | 强基线 |
| +M1 | 仅多尺度融合 | 其余同 Base | 隔离表征改造 |
| +M2 | 仅邻域一致性评分 | 其余同 Base | 隔离 patch score 改造 |
| +M3 | 仅一致性聚合 | 其余同 Base | 隔离偏移适配 |
| Full | M1+M2+M3 | 其余同 Base | 验证组合收益 |

正式实验使用 15 个 MVTec AD 类别、seed=17/29/41，保留每个类别的正常训练/验证划分；若 smoke 通过，再扩展到 5 seed。

## 评价指标

- image-level AUROC：主指标；
- pixel-level AUROC 与 PRO：缺陷定位指标；
- 每类别结果和 macro mean；
- 配对 bootstrap 95% CI、paired t-test、Wilcoxon；
- 特征提取耗时、推理耗时、显存和 memory bank 大小。

## 成功与停止规则

- 只有当 Full 相对 Base 的主指标提升具有不跨 0 的配对 CI，才写成稳定收益；
- 若仅某一缺陷族提升，写成条件性收益；
- 若 Full 不优于 Base，保留负结果，不通过调参反复筛选结果；
- 任何模块不得使用测试异常标签进行阈值或 memory bank 选择。

## FullDual 补充确认

由于首轮结果出现“Full 的 pixel/PRO 改善与 image AUROC 下降”的任务冲突，增加一个预先固定的双评分变体。FullDual 使用 Base 与 Full 的训练正常 patch-score 分布分别做 z-score 校准，固定权重 0.5/0.5 融合得到 image-level score；pixel-level AUROC 与 PRO 仍只使用 Full 的定位图。融合权重不使用测试异常标签调节，结果仅用于验证任务解耦是否能消除图像级负迁移。

## 资源与复现

- 数据源：`/root/sj-tmp/rata_027/data/mvtec_ad`；
- 独立运行目录：`/root/sj-fs/exp050/runs/mvtec-patchcore-v2`；
- 预训练权重：torchvision WideResNet50-2；
- 固定环境、seed、类别顺序、正常图像划分和 memory budget；
- 每次运行保存配置、类别级 JSON、汇总 CSV、资源记录和代码哈希。
