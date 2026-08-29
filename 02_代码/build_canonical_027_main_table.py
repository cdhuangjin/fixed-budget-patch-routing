#!/usr/bin/env python3
"""Build a canonical 027 main table across MVTec/VisA/MPDD with selective-inference metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def ci95(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if len(values) < 2:
        return mean(values), mean(values)
    avg = mean(values)
    se = stdev(values) / math.sqrt(len(values))
    return avg - 1.96 * se, avg + 1.96 * se


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_latency_ms(metrics: dict) -> float | None:
    """Read either the legacy measured field or the v2 explicitly estimated field."""
    return metrics.get("mean_ms", metrics.get("mean_ms_estimated"))


def collect_mvtec(mvtec_root: Path, *, route: str = "risk_fallback") -> list[dict]:
    random_route = {
        "risk_fallback": "random_fallback",
        "strict_quota": "strict_quota_random",
        "online_prefix_quota": "online_prefix_quota_random",
    }.get(route)
    if random_route is None:
        raise ValueError(f"unsupported MVTec route: {route}")
    rows = []
    for seed_dir in sorted(p for p in mvtec_root.iterdir() if p.is_dir() and p.name.startswith("seed")):
        seed = int(seed_dir.name.replace("seed", ""))
        for path in sorted(seed_dir.glob("*.json")):
            data = read_json(path)
            route_metrics = data[route]
            random_metrics = data[random_route]
            row = {
                "dataset": "MVTec",
                "category": data["category"],
                "seed": seed,
                "fallback_budget": data["fallback_budget"],
                "fast_only_auroc": data["fast_only"]["image_auroc"],
                "full_only_auroc": data["full_only"]["image_auroc"],
                "risk_auroc": route_metrics["image_auroc"],
                "random_auroc": random_metrics["image_auroc"],
                "risk_delta": route_metrics["image_auroc"] - random_metrics["image_auroc"],
                "fallback_rate": route_metrics["fallback_rate"],
                "fast_mean_ms": mean_latency_ms(data["fast_only"]),
                "full_mean_ms": mean_latency_ms(data["full_only"]),
                "risk_mean_ms": mean_latency_ms(route_metrics),
            }
            routing = data.get("routing", {})
            risk_route = routing.get(route)
            random_route_record = routing.get(random_route)
            if risk_route is not None and random_route_record is not None:
                row.update({
                    "risk_count": int(risk_route["actual_fallback_count"]),
                    "random_count": int(random_route_record["actual_fallback_count"]),
                    "total": int(data["n_test"]),
                    "route_source": risk_route.get("route_source", ""),
                })
            rows.append(row)
    return rows


def collect_flat(dataset: str, path: Path, seed: int) -> list[dict]:
    data = read_json(path)
    rows = []
    for item in data:
        rows.append({
            "dataset": dataset,
            "category": item["category"],
            "seed": item.get("seed", seed),
            "fallback_budget": item["budget"],
            "fast_only_auroc": item["fast_only_auroc"],
            "risk_auroc": item["risk_combined_auroc"],
            "random_auroc": item["random_combined_auroc"],
            "risk_delta": item["risk_delta"],
            "fallback_rate": item["fallback_rate"],
            "fast_mean_ms": None,
            "risk_mean_ms": None,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    def group(items: list[dict]) -> dict:
        risk = [item["risk_auroc"] for item in items]
        fast = [item["fast_only_auroc"] for item in items]
        delta = [item["risk_delta"] for item in items]
        fallback = [item["fallback_rate"] for item in items]
        return {
            "count": len(items),
            "risk_auroc_mean": mean(risk),
            "risk_auroc_ci": list(ci95(risk)),
            "fast_auroc_mean": mean(fast),
            "risk_delta_mean": mean(delta),
            "risk_delta_ci": list(ci95(delta)),
            "fallback_rate_mean": mean(fallback),
            "risk_gt_fast_count": sum(1 for r, f in zip(risk, fast) if r >= f - 1e-9),
            "risk_gt_fast_rate": sum(1 for r, f in zip(risk, fast) if r >= f - 1e-9) / len(items),
        }

    overall = group(rows)
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        by_dataset[dataset] = group([row for row in rows if row["dataset"] == dataset])

    by_budget_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        by_budget_dataset[dataset] = {}
        for budget in sorted({row["fallback_budget"] for row in rows if row["dataset"] == dataset}):
            subset = [row for row in rows if row["dataset"] == dataset and row["fallback_budget"] == budget]
            by_budget_dataset[dataset][str(budget)] = group(subset)

    return {
        "overall": overall,
        "by_dataset": by_dataset,
        "by_budget_dataset": by_budget_dataset,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec-dir", required=True, help="MVTec multiseed results (seed dirs with category JSON)")
    parser.add_argument("--visa-json", required=True, help="VisA results JSON (array)")
    parser.add_argument("--mpdd-json", required=True, help="MPDD results JSON (array)")
    parser.add_argument("--mvtec-route", choices=["risk_fallback", "strict_quota", "online_prefix_quota"], default="risk_fallback")
    parser.add_argument("--visa-seed", type=int, default=5)
    parser.add_argument("--mpdd-seed", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    rows.extend(collect_mvtec(Path(args.mvtec_dir), route=args.mvtec_route))
    rows.extend(collect_flat("VisA", Path(args.visa_json), args.visa_seed))
    rows.extend(collect_flat("MPDD", Path(args.mpdd_json), args.mpdd_seed))

    summary = summarize(rows)
    canonical = {
        "status": "canonical_main_table",
        "task": "selective_inference_budget_sensitive",
        "mvtec_route": args.mvtec_route,
        "rows": rows,
        "summary": summary,
        "top_tier_notes": [
            "当前主表以 risk_fallback / risk_combined 的 image_auroc 为核心，并补充 fast-only 与 risk-random delta。",
            "027 的顶刊叙事应围绕同预算下的 risk>random、跨数据集泛化、以及延迟效率展开。",
            "下一步建议在主表补充 strict_quota 下的 delta，以强化选择性推理预算敏感性。"
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "rows": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "overall_risk_gt_fast_rate": summary["overall"]["risk_gt_fast_rate"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
