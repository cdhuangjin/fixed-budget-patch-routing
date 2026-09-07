#!/usr/bin/env python3
"""Materialize audit-ready CSV for the MVTec budget-sensitivity canonical source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


RAW_FIELDS = [
    "dataset", "category", "seed", "fallback_budget",
    "fast_only_auroc", "risk_auroc", "random_auroc", "fast_score_auroc",
    "uncertainty_auroc", "fast_only_recall", "risk_recall", "random_recall",
    "fast_score_recall", "uncertainty_recall", "risk_delta", "risk_recall_delta",
    "risk_minus_fast_score", "risk_minus_uncertainty", "fallback_rate",
    "fallback_count", "total",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _audit(canonical: dict) -> dict:
    rows = canonical["rows"]
    checks = []

    expected_counts = {f"{budget:.2f}": 45 for budget in canonical["protocol"]["fallback_budgets"]}
    observed_counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['fallback_budget']:.2f}"
        observed_counts[key] = observed_counts.get(key, 0) + 1
    checks.append({
        "name": "budget_row_counts",
        "status": "PASS" if observed_counts == expected_counts else "FAIL",
        "detail": f"observed={observed_counts}, expected={expected_counts}",
    })

    matched = all(
        abs(row["fallback_rate"] - row["fallback_count"] / row["total"]) < 1e-12
        and row["fallback_count"] == math.ceil(row["total"] * row["fallback_budget"] - 1e-12)
        for row in rows
    )
    checks.append({
        "name": "matched_exact_quota",
        "status": "PASS" if matched else "FAIL",
        "detail": "all rows enforce ceil(n * budget)",
    })

    arithmetic = all(
        abs(row["risk_delta"] - (row["risk_auroc"] - row["random_auroc"])) < 1e-12
        and abs(row["risk_recall_delta"] - (row["risk_recall"] - row["random_recall"])) < 1e-12
        and abs(row["risk_minus_fast_score"] - (row["risk_auroc"] - row["fast_score_auroc"])) < 1e-12
        and abs(row["risk_minus_uncertainty"] - (row["risk_auroc"] - row["uncertainty_auroc"])) < 1e-12
        for row in rows
    )
    checks.append({
        "name": "budget_baseline_arithmetic",
        "status": "PASS" if arithmetic else "FAIL",
        "detail": "all deltas equal Risk minus route baseline",
    })

    seeds = canonical["protocol"]["seeds"]
    budget_labels = {f"{b:.2f}" for b in canonical["protocol"]["fallback_budgets"]}
    per_seed = all(
        {
            row["seed"] for row in rows
            if f"{row['fallback_budget']:.2f}" == label
        } == set(seeds)
        for label in budget_labels
    )
    checks.append({
        "name": "seed_layout",
        "status": "PASS" if per_seed else "FAIL",
        "detail": f"each budget has seeds {seeds}",
    })

    fail_count = sum(check["status"] == "FAIL" for check in checks)
    return {
        "summary": {"pass": len(checks) - fail_count, "warn": 0, "fail": fail_count},
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    if canonical.get("schema_version") != "canonical_v3_budget_sensitivity_v1":
        raise ValueError("expected canonical_v3_budget_sensitivity_v1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = canonical["rows"]
    _write_csv(args.output_dir / "raw_results.csv", RAW_FIELDS, raw_rows)
    _write_csv(
        args.output_dir / "budget_sensitivity.csv",
        ["budget", "auroc_delta_mean", "auroc_ci_low", "auroc_ci_high",
         "recall_delta_mean", "recall_ci_low", "recall_ci_high",
         "auroc_positive", "recall_positive", "seeds", "unit_count"],
        [
            {
                "budget": budget,
                "auroc_delta_mean": entry["auroc_delta_mean"],
                "auroc_ci_low": entry["auroc_delta_ci"][0],
                "auroc_ci_high": entry["auroc_delta_ci"][1],
                "recall_delta_mean": entry["recall_delta_mean"],
                "recall_ci_low": entry["recall_delta_ci"][0],
                "recall_ci_high": entry["recall_delta_ci"][1],
                "auroc_positive": entry["auroc_positive_count"],
                "recall_positive": entry["recall_positive_count"],
                "seeds": len(canonical["protocol"]["seeds"]),
                "unit_count": entry["unit_count"],
            }
            for budget, entry in sorted(canonical["summary"].items(), key=lambda kv: float(kv[0]))
        ],
    )

    audit = _audit(canonical)
    source_bytes = args.canonical.read_bytes()
    manifest = {
        "schema_version": "budget_sensitivity_materialized_v1",
        "source": str(args.canonical),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_bytes": len(source_bytes),
        "row_count": len(raw_rows),
        "protocol": canonical["protocol"],
        "audit_summary": audit["summary"],
    }
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
