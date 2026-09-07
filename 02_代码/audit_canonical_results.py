#!/usr/bin/env python3
"""Audit Canonical V2 completeness and internal consistency before writing claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_UNITS = {"MVTec": 75, "MPDD": 30, "VisA": 60}
STRONG_ROUTING_EXPECTED_UNITS = {"MVTec": 45, "MPDD": 18, "VisA": 36}
BUDGETS = (0.10, 0.25, 0.50)
BUDGET_SEEDS = {"5", "17", "29"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _report(checks: list[dict[str, str]]) -> dict[str, Any]:
    summary = {status.lower(): sum(check["status"] == status for check in checks) for status in ("PASS", "WARN", "FAIL")}
    return {"status": "FAIL" if summary["fail"] else "WARN" if summary["warn"] else "PASS", "summary": summary, "checks": checks}


def _close(actual: str, expected: float) -> bool:
    return math.isclose(float(actual), expected, abs_tol=1e-12)


def _audit_strong_routing(canonical_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    raw_rows = _read_csv(canonical_dir / "raw_results.csv")
    main_rows = _read_csv(canonical_dir / "main_results.csv")
    checks: list[dict[str, str]] = []

    checks.append(_check(
        "manifest_row_count",
        "PASS" if int(manifest.get("row_count", -1)) == len(raw_rows) else "FAIL",
        f"manifest={manifest.get('row_count')}, raw={len(raw_rows)}",
    ))
    observed_counts = Counter(row["dataset"] for row in raw_rows)
    checks.append(_check(
        "dataset_layout",
        "PASS" if observed_counts == Counter(STRONG_ROUTING_EXPECTED_UNITS) else "FAIL",
        f"observed={dict(sorted(observed_counts.items()))}; expected={STRONG_ROUTING_EXPECTED_UNITS}",
    ))
    keys = [(row["dataset"], row["category"], row["seed"]) for row in raw_rows]
    duplicate_count = len(keys) - len(set(keys))
    checks.append(_check(
        "unique_category_seed_rows",
        "PASS" if duplicate_count == 0 else "FAIL",
        "all category-seed keys are unique" if duplicate_count == 0 else f"{duplicate_count} duplicate keys",
    ))
    category_seeds = defaultdict(set)
    for row in raw_rows:
        category_seeds[(row["dataset"], row["category"])].add(row["seed"])
    fixed_budget_and_seed_layout = all(
        _close(row["fallback_budget"], 0.25) for row in raw_rows
    ) and all(seeds == BUDGET_SEEDS for seeds in category_seeds.values())
    checks.append(_check(
        "fixed_budget_and_seed_layout",
        "PASS" if fixed_budget_and_seed_layout else "FAIL",
        "all rows use budget 0.25 and every category has seeds [5, 17, 29]"
        if fixed_budget_and_seed_layout else "budget or category seed layout is invalid",
    ))
    arithmetic_ok = all(
        _close(row["risk_delta"], float(row["risk_auroc"]) - float(row["random_auroc"]))
        and _close(row["risk_minus_fast_score"], float(row["risk_auroc"]) - float(row["fast_score_auroc"]))
        and _close(row["risk_minus_uncertainty"], float(row["risk_auroc"]) - float(row["uncertainty_auroc"]))
        for row in raw_rows
    )
    checks.append(_check(
        "strong_baseline_arithmetic",
        "PASS" if arithmetic_ok else "FAIL",
        "all deltas equal Risk minus route baseline" if arithmetic_ok else "one or more deltas are inconsistent",
    ))
    quota_ok = all(
        _close(row["fallback_rate"], int(row["fallback_count"]) / int(row["total"]))
        and int(row["fallback_count"]) == math.ceil(int(row["total"]) * float(row["fallback_budget"]) - 1e-12)
        for row in raw_rows
    )
    checks.append(_check(
        "matched_exact_quota",
        "PASS" if quota_ok else "FAIL",
        "all rows enforce ceil(n * budget)" if quota_ok else "one or more rows violate the exact quota",
    ))

    main_by_dataset = {row["dataset"]: row for row in main_rows}
    main_errors: list[str] = []
    mean_columns = {
        "fast_only_auroc_mean": "fast_only_auroc",
        "risk_auroc_mean": "risk_auroc",
        "random_auroc_mean": "random_auroc",
        "fast_score_auroc_mean": "fast_score_auroc",
        "uncertainty_auroc_mean": "uncertainty_auroc",
        "risk_random_delta_mean": "risk_delta",
        "risk_fast_score_delta_mean": "risk_minus_fast_score",
        "risk_uncertainty_delta_mean": "risk_minus_uncertainty",
    }
    if len(main_rows) != len(STRONG_ROUTING_EXPECTED_UNITS) or set(main_by_dataset) != set(STRONG_ROUTING_EXPECTED_UNITS):
        main_errors.append("summary rows")
    for dataset, expected_count in STRONG_ROUTING_EXPECTED_UNITS.items():
        rows = [row for row in raw_rows if row["dataset"] == dataset]
        main = main_by_dataset.get(dataset)
        if main is None or int(main["unit_count"]) != expected_count:
            main_errors.append(dataset)
            continue
        if any(not _close(main[main_column], sum(float(row[raw_column]) for row in rows) / len(rows))
               for main_column, raw_column in mean_columns.items()):
            main_errors.append(dataset)
    checks.append(_check(
        "main_table_aggregation",
        "PASS" if not main_errors else "FAIL",
        "dataset aggregates match raw rows" if not main_errors else f"mismatch: {', '.join(main_errors)}",
    ))
    return _report(checks)


def _audit_budget_sensitivity(canonical_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    raw_rows = _read_csv(canonical_dir / "raw_results.csv")
    summary_rows = _read_csv(canonical_dir / "budget_sensitivity.csv")
    checks: list[dict[str, str]] = []

    checks.append(_check(
        "manifest_row_count",
        "PASS" if int(manifest.get("row_count", -1)) == len(raw_rows) == 135 else "FAIL",
        f"manifest={manifest.get('row_count')}, raw={len(raw_rows)}, expected=135",
    ))
    expected_counts = {f"{budget:.2f}": 45 for budget in BUDGETS}
    observed_counts = Counter(f"{float(row['fallback_budget']):.2f}" for row in raw_rows)
    checks.append(_check(
        "budget_row_counts",
        "PASS" if observed_counts == Counter(expected_counts) else "FAIL",
        f"observed={dict(sorted(observed_counts.items()))}; expected={expected_counts}",
    ))
    observed_seeds = {
        budget: {row["seed"] for row in raw_rows if f"{float(row['fallback_budget']):.2f}" == budget}
        for budget in expected_counts
    }
    checks.append(_check(
        "seed_layout",
        "PASS" if all(seeds == BUDGET_SEEDS for seeds in observed_seeds.values()) else "FAIL",
        f"observed={observed_seeds}; expected={sorted(BUDGET_SEEDS)}",
    ))
    categories_by_budget = {
        budget: defaultdict(set)
        for budget in expected_counts
    }
    for row in raw_rows:
        budget = f"{float(row['fallback_budget']):.2f}"
        if budget in categories_by_budget:
            categories_by_budget[budget][row["category"]].add(row["seed"])
    dataset_category_seed_layout = (
        all(row["dataset"] == "MVTec" for row in raw_rows)
        and all(
            len(categories) == 15 and all(seeds == BUDGET_SEEDS for seeds in categories.values())
            for categories in categories_by_budget.values()
        )
    )
    checks.append(_check(
        "budget_dataset_category_seed_layout",
        "PASS" if dataset_category_seed_layout else "FAIL",
        "MVTec-only with 15 categories and three seeds at each budget"
        if dataset_category_seed_layout else "dataset, category, or seed layout is invalid",
    ))
    keys = [(row["dataset"], row["category"], row["seed"], f"{float(row['fallback_budget']):.2f}") for row in raw_rows]
    duplicate_count = len(keys) - len(set(keys))
    checks.append(_check(
        "unique_category_seed_budget_rows",
        "PASS" if duplicate_count == 0 else "FAIL",
        "all category-seed-budget keys are unique" if duplicate_count == 0 else f"{duplicate_count} duplicate keys",
    ))
    arithmetic_ok = all(
        _close(row["risk_delta"], float(row["risk_auroc"]) - float(row["random_auroc"]))
        and _close(row["risk_recall_delta"], float(row["risk_recall"]) - float(row["random_recall"]))
        and _close(row["risk_minus_fast_score"], float(row["risk_auroc"]) - float(row["fast_score_auroc"]))
        and _close(row["risk_minus_uncertainty"], float(row["risk_auroc"]) - float(row["uncertainty_auroc"]))
        for row in raw_rows
    )
    checks.append(_check(
        "budget_baseline_arithmetic",
        "PASS" if arithmetic_ok else "FAIL",
        "all deltas equal Risk minus route baseline" if arithmetic_ok else "one or more deltas are inconsistent",
    ))
    quota_ok = all(
        _close(row["fallback_rate"], int(row["fallback_count"]) / int(row["total"]))
        and int(row["fallback_count"]) == math.ceil(int(row["total"]) * float(row["fallback_budget"]) - 1e-12)
        for row in raw_rows
    )
    checks.append(_check(
        "matched_exact_quota",
        "PASS" if quota_ok else "FAIL",
        "all rows enforce ceil(n * budget)" if quota_ok else "one or more rows violate the exact quota",
    ))

    summary_by_budget = {f"{float(row['budget']):.2f}": row for row in summary_rows}
    summary_errors: list[str] = []
    if len(summary_rows) != len(expected_counts) or set(summary_by_budget) != set(expected_counts):
        summary_errors.append("summary rows")
    for budget in expected_counts:
        rows = [row for row in raw_rows if f"{float(row['fallback_budget']):.2f}" == budget]
        summary = summary_by_budget.get(budget)
        if summary is None or int(summary["unit_count"]) != len(rows) or int(summary["seeds"]) != len(BUDGET_SEEDS):
            summary_errors.append(budget)
            continue
        if (
            not _close(summary["auroc_delta_mean"], sum(float(row["risk_delta"]) for row in rows) / len(rows))
            or not _close(summary["recall_delta_mean"], sum(float(row["risk_recall_delta"]) for row in rows) / len(rows))
            or int(summary["auroc_positive"]) != sum(float(row["risk_delta"]) > 0 for row in rows)
            or int(summary["recall_positive"]) != sum(float(row["risk_recall_delta"]) > 0 for row in rows)
        ):
            summary_errors.append(budget)
    checks.append(_check(
        "budget_summary_aggregation",
        "PASS" if not summary_errors else "FAIL",
        "budget summary aggregates match raw rows" if not summary_errors else f"mismatch: {', '.join(summary_errors)}",
    ))
    return _report(checks)


def audit_canonical_results(
    canonical_dir: Path,
    *,
    expected_units: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a PASS/WARN/FAIL audit of a materialized Canonical V2 directory."""
    manifest_path = canonical_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = manifest.get("schema_version")
        if schema_version == "strong_routing_materialized_v1":
            return _audit_strong_routing(canonical_dir, manifest)
        if schema_version == "budget_sensitivity_materialized_v1":
            return _audit_budget_sensitivity(canonical_dir, manifest)

    expected_units = expected_units or DEFAULT_EXPECTED_UNITS
    checks: list[dict[str, str]] = []
    required = [
        "raw_results.csv", "routing_results.csv", "main_results.csv", "stats_results.csv",
        "ablation_results.csv", "efficiency_results.csv", "manifest.json",
    ]
    missing = [name for name in required if not (canonical_dir / name).is_file()]
    checks.append(_check(
        "required_materialized_files",
        "PASS" if not missing else "FAIL",
        "all required files are present" if not missing else f"missing: {', '.join(missing)}",
    ))
    if missing:
        return {"status": "FAIL", "summary": {"pass": 0, "warn": 0, "fail": 1}, "checks": checks}

    raw_rows = _read_csv(canonical_dir / "raw_results.csv")
    main_rows = _read_csv(canonical_dir / "main_results.csv")
    stats_rows = _read_csv(canonical_dir / "stats_results.csv")
    routing_rows = _read_csv(canonical_dir / "routing_results.csv")
    manifest = json.loads((canonical_dir / "manifest.json").read_text(encoding="utf-8"))

    checks.append(_check(
        "raw_result_rows",
        "PASS" if raw_rows else "FAIL",
        f"{len(raw_rows)} rows" if raw_rows else "raw_results.csv is empty",
    ))
    if not raw_rows:
        return {"status": "FAIL", "summary": {"pass": 1, "warn": 0, "fail": 1}, "checks": checks}

    row_count_matches = int(manifest.get("source_row_count", -1)) == len(raw_rows)
    checks.append(_check(
        "manifest_source_row_count",
        "PASS" if row_count_matches else "FAIL",
        f"manifest={manifest.get('source_row_count')}, raw={len(raw_rows)}",
    ))

    observed_counts = Counter(row["dataset"] for row in raw_rows)
    layout_ok = observed_counts == Counter(expected_units)
    checks.append(_check(
        "expected_dataset_seed_layout",
        "PASS" if layout_ok else "FAIL",
        f"observed={dict(sorted(observed_counts.items()))}; expected={expected_units}",
    ))

    keys = [(row["dataset"], row["category"], row["seed"]) for row in raw_rows]
    duplicate_count = len(keys) - len(set(keys))
    checks.append(_check(
        "unique_category_seed_rows",
        "PASS" if duplicate_count == 0 else "FAIL",
        "all category-seed keys are unique" if duplicate_count == 0 else f"{duplicate_count} duplicate keys",
    ))

    delta_errors = [
        row for row in raw_rows
        if not math.isclose(float(row["risk_delta"]), float(row["risk_auroc"]) - float(row["random_auroc"]), abs_tol=1e-12)
    ]
    checks.append(_check(
        "risk_delta_arithmetic",
        "PASS" if not delta_errors else "FAIL",
        "all risk deltas match Risk minus Random" if not delta_errors else f"{len(delta_errors)} mismatched deltas",
    ))

    provenance_errors = [
        row for row in raw_rows
        if any(not str(row.get(key, "")).strip() for key in ("risk_count", "random_count", "total"))
    ]
    count_errors = [
        row for row in raw_rows
        if row not in provenance_errors and int(row["risk_count"]) != int(row["random_count"])
    ]
    route_match_errors = [row for row in routing_rows if row.get("matched_counts") != "True"]
    checks.append(_check(
        "matched_fallback_counts",
        "PASS" if not provenance_errors and not count_errors and not route_match_errors else "FAIL",
        "all Risk/Random fallback counts match" if not provenance_errors and not count_errors and not route_match_errors else "missing or unmatched fallback count found",
    ))

    rate_errors = [
        row for row in raw_rows if row not in provenance_errors
        if not math.isclose(float(row["fallback_rate"]), int(row["risk_count"]) / int(row["total"]), abs_tol=1e-12)
    ]
    checks.append(_check(
        "realized_fallback_rates",
        "PASS" if not rate_errors else "FAIL",
        "all realized rates match counts" if not rate_errors else f"{len(rate_errors)} rate/count mismatches",
    ))

    main_by_dataset = {row["dataset"]: row for row in main_rows}
    main_errors = []
    for dataset, dataset_rows in defaultdict(list, {name: [row for row in raw_rows if row["dataset"] == name] for name in observed_counts}).items():
        main = main_by_dataset.get(dataset)
        if main is None:
            main_errors.append(f"missing {dataset}")
            continue
        expected_delta = sum(float(row["risk_delta"]) for row in dataset_rows) / len(dataset_rows)
        if int(main["unit_count"]) != len(dataset_rows) or not math.isclose(float(main["mean_risk_minus_random_auroc"]), expected_delta, abs_tol=1e-12):
            main_errors.append(dataset)
    checks.append(_check(
        "main_table_aggregation",
        "PASS" if not main_errors else "FAIL",
        "dataset aggregates match raw rows" if not main_errors else f"mismatch: {', '.join(main_errors)}",
    ))

    stats_by_dataset = {row["dataset"]: row for row in stats_rows}
    stats_errors = []
    for dataset, main in main_by_dataset.items():
        stat = stats_by_dataset.get(dataset)
        if stat is None:
            stats_errors.append(f"missing {dataset}")
            continue
        mean_delta = float(main["mean_risk_minus_random_auroc"])
        if not math.isclose(float(stat["mean_delta"]), mean_delta, abs_tol=1e-12):
            stats_errors.append(f"mean {dataset}")
        elif not float(stat["ci95_low"]) <= mean_delta <= float(stat["ci95_high"]):
            stats_errors.append(f"CI {dataset}")
    checks.append(_check(
        "statistics_consistency",
        "PASS" if not stats_errors else "FAIL",
        "reported means and CIs agree with main table" if not stats_errors else f"mismatch: {', '.join(stats_errors)}",
    ))

    ablation_rows = _read_csv(canonical_dir / "ablation_results.csv")
    ablation_available = ablation_rows and ablation_rows[0].get("status") != "NOT_AVAILABLE"
    checks.append(_check(
        "strict_quota_ablation_evidence",
        "PASS" if ablation_available else "WARN",
        "structured ablation evidence present" if ablation_available else "no structured strict-quota ablation evidence",
    ))

    efficiency_rows = _read_csv(canonical_dir / "efficiency_results.csv")
    if efficiency_rows and "path" in efficiency_rows[0]:
        expected_paths = {"fast", "full", "risk"}
        paths = {row.get("path") for row in efficiency_rows}
        protocol_ok = all(
            row.get("batch_size") == "1"
            and row.get("cuda_synchronize") == "True"
            and row.get("comparison_scope") == "separate_system_audit_not_paired_with_accuracy_rows"
            for row in efficiency_rows
        )
        checks.append(_check(
            "separate_latency_audit",
            "PASS" if paths == expected_paths and protocol_ok else "FAIL",
            "batch-one CUDA audit remains separate from accuracy rows" if paths == expected_paths and protocol_ok else "latency audit paths or protocol are incomplete",
        ))
    else:
        checks.append(_check("separate_latency_audit", "WARN", "no structured audited latency evidence"))

    return _report(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_canonical_results(args.canonical_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
