# 027 Claim–Evidence Matrix v3

| Claim | Evidence | Status | Boundary |
|---|---|---|---|
| 局部 Patch 路由优于严格 matched Random | 45 个 MVTec、18 个 MPDD、36 个 VisA 类别—seed 单元；25% 配额 AUROC 差值分别 +0.0783、+0.1013、+0.0124，均有正 bootstrap CI | supported | 仅限当前 Fast/Full 特征配置与严格 25% 协议 |
| 收益跨预算 | MVTec 15 类 seed=17；10/25/50% 配额 AUROC 差异 +0.0560/+0.0783/+0.1134 | supported | 10% 与 50% 为单种子扩展，25% 为三种子正式结果 |
| Full 是质量上限 | MVTec Full AUROC 0.9390，高于严格 25% 局部 Patch 路由 0.7758 | supported | 不代表部署延迟上限 |
| PaDiM-style 基线已补充 | 15 类、3 seed、45 单元 AUROC mean 0.8982，Recall mean 0.6544 | supported | 为对角高斯实现，不等同于完整官方实现 |
| 路由机制消融 | score-ranked matched 与 Risk 完全一致；oracle upper bound AUROC/Recall=0.9718/0.9257 | supported | score-ranked 是当前 Risk 的等价实现，oracle 不进入主结论 |
| 局部探针降低逐样本 P95 | 局部探针 P95 3.127 ms，高于 Fast 2.995 ms但低于 Full 4.714 ms；混合路径 P95 7.135 ms | unsupported | 不能主张混合路径端到端加速 |
| 局部探针具有固定预算质量收益 | 三数据集严格 25% matched-random、99 个类别—seed 单元 | supported | 只能主张预算分配收益，不主张全面精度领先 |
| Full 具备定位能力 | pixel-AUROC mean 0.9543 | supported | 不等价于 Risk 路由定位能力 |
| Full 路径具备区域定位能力 | seed=17、15 类 pixel-AUROC mean 0.9543，PRO@FPR=0.30 mean 0.8380 | supported | 仅限 Full 定位审计，不等价于 Risk 路由定位收益 |
| 局部探针路由改善定位 | 当前仅有 Full 路径定位审计，无路由定位覆盖率 | unsupported | 不在主文宣称路由定位收益 |
| 跨工业场景泛化 | MPDD 与 VisA 的严格 25% matched-random 外部验证 | supported with boundary | 外部数据集类别数较少，不能宣称普遍泛化 |
| 风险排序机制优于其他门控 | 严格 matched-random 与旧全局 Risk 的负对照；局部 layer-2 patch score 改变排序 | supported with boundary | 尚缺 learned uncertainty 等更强门控基线 |
