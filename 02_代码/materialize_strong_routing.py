#!/usr/bin/env python3
"""Materialize audit-ready CSV tables for the strong-routing canonical table."""

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
    "fast_score_recall", "uncertainty_recall", "risk_delta", "risk_minus_fast_score",
    "risk_minus_uncertainty", "fallback_rate", "fallback_count", "total",
]
SUMMARY_FIELDS = [
    "dataset", "unit_count", "fast_only_auroc_mean", "risk_auroc_mean",
    "random_auroc_mean", "fast_score_auroc_mean", "uncertainty_auroc_mean",
    "risk_random_delta_mean", "risk_random_delta_low", "risk_random_delta_high",
    "risk_fast_score_delta_mean", "risk_fast_score_delta_low",
    "risk_fast_score_delta_high", "risk_uncertainty_delta_mean",
    "risk_uncertainty_delta_low", "risk_uncertainty_delta_high",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _dataset_summary(rows: list[dict], summary: dict) -> list[dict]:
    output = []
    for dataset in ("MVTec", "MPDD", "VisA"):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        comparisons = summary["by_dataset"][dataset]
        row = {
            "dataset": dataset,
            "unit_count": len(dataset_rows),
            "fast_only_auroc_mean": sum(r["fast_only_auroc"] for r in dataset_rows) / len(dataset_rows),
            "risk_auroc_mean": sum(r["risk_auroc"] for r in dataset_rows) / len(dataset_rows),
            "random_auroc_mean": sum(r["random_auroc"] for r in dataset_rows) / len(dataset_rows),
            "fast_score_auroc_mean": sum(r["fast_score_auroc"] for r in dataset_rows) / len(dataset_rows),
            "uncertainty_auroc_mean": sum(r["uncertainty_auroc"] for r in dataset_rows) / len(dataset_rows),
        }
        for name in ("risk_random", "risk_fast_score", "risk_uncertainty"):
            comparison = comparisons[name]
            row[f"{name}_delta_mean"] = comparison["delta_mean"]
            row[f"{name}_delta_low"] = comparison["delta_ci"][0]
            row[f"{name}_delta_high"] = comparison["delta_ci"][1]
        output.append(row)
    return output


def _audit(canonical: dict) -> dict:
    rows = canonical["rows"]
    checks = []
    checks.append({
        "name": "expected_row_count",
        "status": "PASS" if len(rows) == 99 else "FAIL",
        "detail": f"observed={len(rows)}, expected=99",
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
        and abs(row["risk_minus_fast_score"] - (row["risk_auroc"] - row["fast_score_auroc"])) < 1e-12
        and abs(row["risk_minus_uncertainty"] - (row["risk_auroc"] - row["uncertainty_auroc"])) < 1e-12
        for row in rows
    )
    checks.append({
        "name": "strong_baseline_arithmetic",
        "status": "PASS" if arithmetic else "FAIL",
        "detail": "all deltas equal Risk minus route baseline",
    })
    layout = {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in ("MVTec", "MPDD", "VisA")}
    checks.append({
        "name": "dataset_layout",
        "status": "PASS" if layout == {"MVTec": 45, "MPDD": 18, "VisA": 36} else "FAIL",
        "detail": str(layout),
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
    if canonical.get("schema_version") != "canonical_v3_strong_routing_v1":
        raise ValueError("expected canonical_v3_strong_routing_v1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = canonical["rows"]
    _write_csv(args.output_dir / "raw_results.csv", RAW_FIELDS, raw_rows)
    dataset_rows = _dataset_summary(raw_rows, canonical["summary"])
    _write_csv(args.output_dir / "main_results.csv", SUMMARY_FIELDS, dataset_rows)
    _write_csv(
        args.output_dir / "category_results.csv",
        ["dataset", "category", "fast_only_auroc_mean", "risk_auroc_mean",
         "random_auroc_mean", "fast_score_auroc_mean", "uncertainty_auroc_mean"],
        canonical["summary"]["category_aggregate"],
    )

    audit = _audit(canonical)
    source_bytes = args.canonical.read_bytes()
    manifest = {
        "schema_version": "strong_routing_materialized_v1",
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
