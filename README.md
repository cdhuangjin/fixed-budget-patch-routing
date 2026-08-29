# 实验027：固定推理预算下的风险感知选择性计算（V2 重构 + V3 Strong-Baseline）

本仓库已按《`027_拒稿后重构_固定预算风险感知选择性计算_0到1完整执行方案.md`》完成目录重构，项目路径为 `06_主线项目/027_fixed_budget_v2/`。

## 重构入口
- 方案文件：`027_拒稿后重构_固定预算风险感知选择性计算_0到1完整执行方案.md`
- 重构说明：`docs/restructuring/README.md`
- Canonical V2 目录：`05_运行记录/canonical_v2/`
- Canonical V3 目录（当前路由结果）：`05_运行记录/canonical_v3/`
- Legacy V1 目录：`05_运行记录/legacy_v1/`
- Legacy 资产索引：`docs/restructuring/legacy_asset_index.md`
- Formal V2 目录：`05_运行记录/formal_v2/`
- Formal V3 目录：`05_运行记录/formal_v3/`
- 分析与图表目录：`06_分析与图表/`
- 论文 V2 目录：`07_paper/`
- 当前投稿包（SIViP）：`07_paper/submission/sivip_submission/`

## 当前进展
- 已建立 canonical_v2 骨架（header-only CSV + manifest.json）
- 已将 V1 canonical/stats 等权威文件迁移到 `legacy_v1/`
- 已将论文、图表、投稿材料迁移到 `07_paper/`
- 已完成目录到方案目标的映射说明
- 已建立 `07_paper/`、`06_分析与图表/tables|failure_cases` 的 V2 子结构
- 已完成 V3 strong-routing 三种子基线（Fast-score / Uncertainty / matched Random），并合并到 `formal_v3/strong_routing_canonical_v1.json`
- 已物化 `canonical_v3/`（raw/main/category CSV + audit_report + manifest）并生成图 `06_分析与图表/canonical_v3/fig_canonical_v3_strong_routing.{pdf,png,svg,tiff}`
- 已把 V3 图复制到 `07_paper/figures/canonical_v3/` 并更新 `main_v2.tex`
- 已建立独立 SIViP 投稿包：期刊元数据、cover letter、可用性声明和上传 ZIP 均已从 PAAA 历史包分离；三份 TeX 源可独立编译

## 建议下一步
1. 作者在公开仓库创建并推送 `sivip_rebuild` 分支，固定含本次重构的 commit
2. 将该公开 commit 写入 SIViP 稿件的 Data/Code availability 声明
3. 在 SIViP 投稿系统中核对作者、资助、利益冲突和数据集许可信息
