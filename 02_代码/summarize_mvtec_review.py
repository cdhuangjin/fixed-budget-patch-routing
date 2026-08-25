"""Aggregate MVTec route JSON files into a reviewer-auditable summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_protocol import paired_bootstrap_delta_ci


def load_results(root):
    paths = sorted(Path(root).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no category JSON files under {root}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _metric(results, method, metric):
    return np.asarray([result[method][metric] for result in results], dtype=float)


def summarize_roots(roots, reps=10000, seed=17):
    summaries = {}
    for root in roots:
        results = load_results(root)
        methods = ("fast_only", "random_fallback", "risk_fallback", "full_only")
        summary = {"n_categories": len(results), "categories": [r["category"] for r in results]}
        for method in methods:
            summary[method] = {
                "image_auroc_mean": float(_metric(results, method, "image_auroc").mean()),
                "recall_at_fpr_mean": float(_metric(results, method, "recall_at_fpr").mean()),
                "fallback_rate_mean": float(_metric(results, method, "fallback_rate").mean()),
            }
        risk_auc = _metric(results, "risk_fallback", "image_auroc")
        random_auc = _metric(results, "random_fallback", "image_auroc")
        risk_recall = _metric(results, "risk_fallback", "recall_at_fpr")
        random_recall = _metric(results, "random_fallback", "recall_at_fpr")
        summary["paired_deltas"] = {
            "auroc_risk_minus_random": paired_bootstrap_delta_ci(risk_auc, random_auc, reps=reps, seed=seed),
            "recall_risk_minus_random": paired_bootstrap_delta_ci(risk_recall, random_recall, reps=reps, seed=seed),
            "auroc_positive_categories": int((risk_auc > random_auc).sum()),
            "recall_positive_categories": int((risk_recall > random_recall).sum()),
        }
        summary["routing_audit"] = [
            {
                "category": result["category"],
                "risk_count": result["routing"]["risk"]["actual_fallback_count"],
                "random_count": result["routing"]["random"]["actual_fallback_count"],
                "counts_match": result["routing"]["risk"]["actual_fallback_count"] == result["routing"]["random"]["actual_fallback_count"],
            }
            for result in results
        ]
        summaries[str(root)] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = summarize_roots(args.roots, reps=args.reps, seed=args.seed)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
