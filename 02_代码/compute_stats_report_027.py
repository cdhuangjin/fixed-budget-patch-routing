#!/usr/bin/env python3
import json
from pathlib import Path
import argparse
from stats_utils import mean, ci95, ttest_paired, cohen_d

def main():
    p=argparse.ArgumentParser(); p.add_argument("--canonical",required=True); p.add_argument("--output",required=True)
    args=p.parse_args()
    canonical=json.loads(Path(args.canonical).read_text(encoding="utf-8"))
    rows=canonical["rows"]

    grouped={}
    for r in rows:
        grouped.setdefault(r["dataset"],{}).setdefault(r["category"],{})[r["seed"]]=r

    analyses=[]
    for dataset in sorted(grouped):
        for category in sorted(grouped[dataset]):
            seeds=grouped[dataset][category]
            skeys=sorted(seeds)
            risk=[seeds[s]["risk_auroc"] for s in skeys]
            rand=[seeds[s]["random_auroc"] for s in skeys]
            fast=[seeds[s]["fast_only_auroc"] for s in skeys]
            t1,p1=ttest_paired(risk,rand)
            t2,p2=ttest_paired(risk,fast)
            analyses.append({"dataset":dataset,"category":category,"n":len(skeys),
              "risk_auroc_mean":mean(risk),"random_auroc_mean":mean(rand),"fast_auroc_mean":mean(fast),
              "risk_gt_random":sum(x>y for x,y in zip(risk,rand)),
              "risk_gt_fast":sum(x>y for x,y in zip(risk,fast)),
              "risk_vs_random":{"delta_mean":mean([x-y for x,y in zip(risk,rand)]),"ci95_delta":list(ci95([x-y for x,y in zip(risk,rand)])),"t_stat":t1,"p_value":p1,"cohen_d":cohen_d(risk,rand)},
              "risk_vs_fast":{"delta_mean":mean([x-y for x,y in zip(risk,fast)]),"ci95_delta":list(ci95([x-y for x,y in zip(risk,fast)])),"t_stat":t2,"p_value":p2,"cohen_d":cohen_d(risk,fast)}})

    dataset_aggregate=[]
    for dataset in sorted(grouped):
        categories=sorted(grouped[dataset])
        risk_means=[]; rand_means=[]; fast_means=[]
        risk_gt_random=0; risk_gt_fast=0; count=0
        for category in categories:
            seed_rows=list(grouped[dataset][category].values())
            risk_means.append(mean([r["risk_auroc"] for r in seed_rows]))
            rand_means.append(mean([r["random_auroc"] for r in seed_rows]))
            fast_means.append(mean([r["fast_only_auroc"] for r in seed_rows]))
            risk_gt_random += sum(r["risk_auroc"] > r["random_auroc"] for r in seed_rows)
            risk_gt_fast += sum(r["risk_auroc"] > r["fast_only_auroc"] for r in seed_rows)
            count += len(seed_rows)
        t1,p1=ttest_paired(risk_means,rand_means)
        t2,p2=ttest_paired(risk_means,fast_means)
        dataset_aggregate.append({"dataset":dataset,"categories":len(categories),"category_means_used":len(categories),
          "risk_gt_random_total":risk_gt_random,"risk_gt_fast_total":risk_gt_fast,"count":count,
          "risk_vs_random":{"delta_mean":mean([x-y for x,y in zip(risk_means,rand_means)]),"ci95_delta":list(ci95([x-y for x,y in zip(risk_means,rand_means)])),"t_stat":t1,"p_value":p1,"cohen_d":cohen_d(risk_means,rand_means)},
          "risk_vs_fast":{"delta_mean":mean([x-y for x,y in zip(risk_means,fast_means)]),"ci95_delta":list(ci95([x-y for x,y in zip(risk_means,fast_means)])),"t_stat":t2,"p_value":p2,"cohen_d":cohen_d(risk_means,fast_means)}})

    summary=canonical["summary"]
    report={"status":"stats_report","project":"027","analyses":analyses,"dataset_aggregate":dataset_aggregate,"summary":summary,"top_tier_notes":[
      "category级分析用于展示细粒度优势；dataset级聚合用于满足期刊常见的主表统计口径。"
    ]}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(out),"risk_gt_random_total":sum(a["risk_gt_random"] for a in analyses),"risk_gt_fast_total":sum(a["risk_gt_fast"] for a in analyses),"datasets":len(dataset_aggregate)},ensure_ascii=False))
if __name__=="__main__": main()

