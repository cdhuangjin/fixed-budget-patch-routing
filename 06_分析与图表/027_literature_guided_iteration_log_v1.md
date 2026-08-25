# 027 文献引导的实验迭代记录

## 检索范围

本轮从本地论文库 `C:\Users\PC\Documents\paper` 选取了与工业异常检测、匹配代价和实时推理直接相关的工作：

- `15_目标检测/ICML2025/2985_CostFilter-AD_Enhancing_Anomaly_Detection_through_Matching_Cost_Filtering.pdf`
- `15_目标检测/ICML2025/0613_Demeaned_Sparse_Efficient_Anomaly_Detection_by_Residual_Estimate.pdf`
- `11_多模态与视觉语言模型/ICML2025/2725_When_Every_Millisecond_Counts_Real-Time_Anomaly_Detection_via_the_Multimodal_Asy.pdf`

## 可迁移的实验思想

CostFilter-AD 强调对输入与正常样本之间的匹配代价进行过滤，而不是直接依赖单一最近邻代价；Demeaned Sparse 强调在无异常标签条件下利用结构残差构造可解释检测信号；实时异常检测工作强调同时报告准确率与响应时间，而不是只报告离线 AUROC。

## 本项目实现的两项探索

1. 最近邻间隔路由：$s=d_1+\alpha(d_2-d_1)$，其中 $\alpha=0.5$。
2. Top-k 匹配代价路由：使用前 5 个正常邻居距离均值作为路由分数。

两项探索均只使用正常校准数据确定阈值，不使用测试标签。VisA candle smoke test 中，间隔路由 AUROC=0.96532、Recall@5%FPR=0.74；完整 VisA seed=17 中，间隔路由相对原 Risk 的 AUROC 差异仅 +0.00028，Recall 差异 +0.00667，提升不稳定。因此两项轻量改造均不纳入主方法，避免把微小波动包装成创新。

## 当前方法学判断

文献对照表明，若要达到顶刊级方法创新，下一阶段需要真正的匹配代价过滤模块或具有理论约束的风险校准，而不是继续堆叠固定邻居统计量。当前结果已经足以形成应用型“预算感知选择性推理”论文，但不应声称已经达到顶刊级通用机制标准。

进一步测试了基于水平翻转一致性的风险信号：以原图和翻转图 Fast 分数的最大值进行正常校准和路由。VisA candle smoke test 中 Risk AUROC=0.5286、Recall@5%FPR=0.03，低于匹配 Random 的 0.5677 和 0.22，判定为失败消融，不纳入主方法。

## 进一步文献复核与严格预算实验

本轮补充复核了本地论文库中的 `5761_Quantifying_Statistical_Significance_of_Deep_Nearest_Neighbor_Anomaly_Detection.pdf`、`2634_What_Does_It_Take_to_Build_a_Performant_Selective_Classifier.pdf`、`3098_Conformal_Anomaly_Detection_in_Event_Sequences.pdf`、`0051_RareCLIP_Rarity-aware_Online_Zero-shot_Industrial_Anomaly_Detection.pdf`。共同启示是：必须区分风险排序质量、统计/分布偏移误差与实际计算预算，不能用名义阈值替代真实预算控制。

据此新增严格配额路由，在 MVTec seed=17 的 15 类上固定选择 25% 样本回退。实际回退率为 25.34%，但 AUROC 相对匹配随机仅 +0.0001，9/15 类为正；说明原 Risk 的 +0.0763 AUROC 优势伴随 72.23% 的实际回退率，不能解释为低预算下的排序优势。

随后测试 128/160 分辨率的多尺度 rank fusion。bottle、cable、capsule 严格配额 smoke test 的 AUROC 均值为 0.7136，低于随机回退的 0.8012，判定为失败改造。当前最可靠结论仍是：Risk 在高实际回退率下有应用价值，但尚未形成严格预算下的顶刊级方法创新。

## 局部 Patch 风险探针与严格 matched-random 重跑

针对全局平均 Fast 特征丢失细粒度缺陷的问题，新增 layer-2 局部 patch 探针。正常 memory bank 按 4 倍下采样后建立，路由分数取局部最近邻距离的 top-5% 均值；严格选择 25% 样本回退 Full。第一轮发现 Random 对照错误复用了旧 Risk 的回退数量，已全部作废并重跑。

严格 matched-random 重跑结果：MVTec 45 单元 AUROC +0.0783（95% CI [+0.0525,+0.1029]），Recall +0.1848（[+0.1192,+0.2491]）；MPDD 18 单元 AUROC +0.1013（[+0.0593,+0.1448]），Recall +0.2074（[+0.1208,+0.3001]）；VisA 36 单元 AUROC +0.0124（[+0.0079,+0.0166]），Recall +0.2283（[+0.1483,+0.3056]）。三数据集共 99 个单元，实际回退率约 25.3%。该结果是当前最有力的改进证据，但仍需真实端到端 P95 审计和与强基线协议对齐。
