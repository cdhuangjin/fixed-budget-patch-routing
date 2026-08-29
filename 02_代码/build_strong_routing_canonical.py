#!/usr/bin/env python3
"""Build the Canonical V3 table for matched strong routing baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analysis_protocol import paired_bootstrap_delta_ci


MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
}
MPDD_CATEGORIES = {
    "bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes",
}
VISA_CATEGORIES = {
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
}
SEEDS = {5, 17, 29}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: float, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: metric is not finite")
    return result


def _validate_counts(context: str, *counts: int) -> int:
    if len(set(counts)) != 1:
        raise ValueError(f"{context}: fallback counts do not match: {counts}")
    return int(counts[0])


def _row(
    dataset: str, item: dict, fallback_count: int, context: str
) -> dict:
    metrics = {
        "fast_only_auroc": item["fast_only_auroc"],
        "fast_only_recall": item["fast_only_recall"],
        "risk_auroc": item["risk_auroc"],
        "risk_recall": item["risk_recall"],
        "random_auroc": item["random_auroc"],
        "random_recall": item["random_recall"],
        "fast_score_auroc": item["fast_score_auroc"],
        "fast_score_recall": item["fast_score_recall"],
        "uncertainty_auroc": item["uncertainty_auroc"],
        "uncertainty_recall": item["uncertainty_recall"],
    }
    metrics = {name: _finite(value, context) for name, value in metrics.items()}
    return {
        "dataset": dataset,
        "category": item["category"],
        "seed": int(item["seed"]),
        "fallback_budget": float(item["fallback_budget"]),
        **metrics,
        "risk_delta": metrics["risk_auroc"] - metrics["random_auroc"],
        "risk_minus_fast_score": metrics["risk_auroc"] - metrics["fast_score_auroc"],
        "risk_minus_uncertainty": metrics["risk_auroc"] - metrics["uncertainty_auroc"],
        "fallback_rate": float(item["fallback_rate"]),
        "fallback_count": fallback_count,
        "total": int(item["total"]),
        "route_sources": item["route_sources"],
    }


def collect_mvtec(root: Path) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        for path in sorted((root / f"seed{seed}").glob("*.json")):
            item = _read_json(path)
            routing = item["routing"]
            count = _validate_counts(
                path,
                routing["strict_quota"]["actual_fallback_count"],
                routing["strict_quota_random"]["actual_fallback_count"],
                routing["fast_score"]["actual_fallback_count"],
                routing["uncertainty_dispersion"]["actual_fallback_count"],
            )
            rows.append(_row(
                "MVTec",
                {
                    "category": item["category"],
                    "seed": item.get("seed", seed),
                    "fallback_budget": item["fallback_budget"],
                    "fallback_rate": item["strict_quota"]["fallback_rate"],
                    "fast_only_auroc": item["fast_only"]["image_auroc"],
                    "fast_only_recall": item["fast_only"]["recall_at_fpr"],
                    "risk_auroc": item["risk_fallback"]["image_auroc"],
                    "risk_recall": item["risk_fallback"]["recall_at_fpr"],
                    "random_auroc": item["strict_quota_random"]["image_auroc"],
                    "random_recall": item["strict_quota_random"]["recall_at_fpr"],
                    "fast_score_auroc": item["fast_score"]["image_auroc"],
                    "fast_score_recall": item["fast_score"]["recall_at_fpr"],
                    "uncertainty_auroc": item["uncertainty_dispersion"]["image_auroc"],
                    "uncertainty_recall": item["uncertainty_dispersion"]["recall_at_fpr"],
                    "total": item["n_test"],
                    "route_sources": {
                        "risk": routing["strict_quota"]["route_source"],
                        "random": routing["strict_quota_random"]["route_source"],
                        "fast_score": routing["fast_score"]["route_source"],
                        "uncertainty": routing["uncertainty_dispersion"]["route_source"],
                    },
                },
                count,
                path,
            ))
    return rows


def collect_external(root: Path) -> list[dict]:
    rows: list[dict] = []
    for disk_name, dataset, expected in (
        ("mpdd", "MPDD", MPDD_CATEGORIES),
        ("visa", "VisA", VISA_CATEGORIES),
    ):
        for path in sorted((root / disk_name).glob("seed*/results.json")):
            for item in _read_json(path):
                count = _validate_counts(
                    path,
                    item["route"]["actual_fallback_count"],
                    item["random_route"]["actual_fallback_count"],
                    item["fast_score_route"]["actual_fallback_count"],
                    item["uncertainty_route"]["actual_fallback_count"],
                )
                rows.append(_row(
                    dataset,
                    {
                        "category": item["category"],
                        "seed": item["seed"],
                        "fallback_budget": item["budget"],
                        "fallback_rate": item["fallback_rate"],
                        "fast_only_auroc": item["fast_only_auroc"],
                        "fast_only_recall": item["fast_only_recall"],
                        "risk_auroc": item["risk_combined_auroc"],
                        "risk_recall": item["risk_combined_recall"],
                        "random_auroc": item["random_combined_auroc"],
                        "random_recall": item["random_combined_recall"],
                        "fast_score_auroc": item["fast_score_combined_auroc"],
                        "fast_score_recall": item["fast_score_combined_recall"],
                        "uncertainty_auroc": item["uncertainty_combined_auroc"],
                        "uncertainty_recall": item["uncertainty_combined_recall"],
                        "total": item["total"],
                        "route_sources": {
                            "risk": item["route"]["route_source"],
                            "random": item["random_route"]["route_source"],
                            "fast_score": item["fast_score_route"]["route_source"],
                            "uncertainty": item["uncertainty_route"]["route_source"],
                        },
                    },
                    count,
                    path,
                ))
    return rows


def _validate_layout(rows: list[dict]) -> None:
    observed: dict[tuple[str, str, int], dict] = {}
    for row in rows:
        key = (row["dataset"], row["category"], row["seed"])
        if key in observed:
            raise ValueError(f"duplicate category-seed row: {key}")
        observed[key] = row
    expected_datasets = {
        "MVTec": (MVTEC_CATEGORIES, 45),
        "MPDD": (MPDD_CATEGORIES, 18),
        "VisA": (VISA_CATEGORIES, 36),
    }
    for dataset, (categories, expected_count) in expected_datasets.items():
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if len(dataset_rows) != expected_count:
            raise ValueError(f"{dataset}: expected {expected_count} rows, got {len(dataset_rows)}")
        missing = categories - {row["category"] for row in dataset_rows}
        extra = {row["category"] for row in dataset_rows} - categories
        if missing or extra:
            raise ValueError(f"{dataset}: category drift; missing={missing}, extra={extra}")
        for category in categories:
            if {row["seed"] for row in dataset_rows if row["category"] == category} != SEEDS:
                raise ValueError(f"{dataset}/{category}: expected seeds {sorted(SEEDS)}")


def _comparison(rows: list[dict], right: str) -> dict:
    left = [row["risk_auroc"] for row in rows]
    other = [row[right] for row in rows]
    bootstrap = paired_bootstrap_delta_ci(left, other, reps=10000, confidence=0.95, seed=17)
    return {
        "risk_auroc_mean": float(sum(left) / len(left)),
        "baseline": right,
        "baseline_auroc_mean": float(sum(other) / len(other)),
        "delta_mean": bootstrap["mean"],
        "delta_ci": [bootstrap["low"], bootstrap["high"]],
        "risk_gt_baseline_count": sum(a > b for a, b in zip(left, other)),
        "unit_count": len(rows),
    }


def summarize(rows: list[dict]) -> dict:
    def aggregate(subset: list[dict]) -> dict:
        return {
            "risk_random": _comparison(subset, "random_auroc"),
            "risk_fast_score": _comparison(subset, "fast_score_auroc"),
            "risk_uncertainty": _comparison(subset, "uncertainty_auroc"),
        }

    by_dataset = {
        dataset: aggregate([row for row in rows if row["dataset"] == dataset])
        for dataset in ("MVTec", "MPDD", "VisA")
    }
    category_aggregate = []
    for dataset in ("MVTec", "MPDD", "VisA"):
        for category in sorted({row["category"] for row in rows if row["dataset"] == dataset}):
            subset = [row for row in rows if row["dataset"] == dataset and row["category"] == category]
            category_aggregate.append({
                "dataset": dataset,
                "category": category,
                "fast_only_auroc_mean": float(sum(row["fast_only_auroc"] for row in subset) / len(subset)),
                "risk_auroc_mean": float(sum(row["risk_auroc"] for row in subset) / len(subset)),
                "random_auroc_mean": float(sum(row["random_auroc"] for row in subset) / len(subset)),
                "fast_score_auroc_mean": float(sum(row["fast_score_auroc"] for row in subset) / len(subset)),
                "uncertainty_auroc_mean": float(sum(row["uncertainty_auroc"] for row in subset) / len(subset)),
            })
    return {
        "overall": aggregate(rows),
        "by_dataset": by_dataset,
        "category_aggregate": category_aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_mvtec(args.mvtec_dir)
    rows.extend(collect_external(args.external_dir))
    _validate_layout(rows)
    canonical = {
        "status": "canonical_strong_routing_main_table",
        "schema_version": "canonical_v3_strong_routing_v1",
        "protocol": {
            "fallback_budget": 0.25,
            "fallback_count_rule": "ceil(n_test * budget)",
            "risk_route": "local patch top-5% mean distance, label-free global exact quota",
            "fast_score_route": "Fast global anomaly score, exact quota matched to Risk",
            "uncertainty_route": "nearest patch-distance Q90-Q50, exact quota matched to Risk",
            "random_control": "seeded uniform sampling, exact quota matched to Risk",
        },
        "rows": rows,
        "summary": summarize(rows),
        "audit_notes": [
            "Risk, Random, Fast-score and Uncertainty use identical fallback counts in every row.",
            "Bootstrap CIs operate on paired category-seed rows and are not independent-dataset inference.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "summary": canonical["summary"]["overall"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
