import csv
import json
from pathlib import Path

from audit_canonical_results import audit_canonical_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_directory(tmp_path, *, duplicate=False):
    directory = tmp_path / "canonical_v2"
    directory.mkdir()
    raw = [
        {"dataset": "MVTec", "category": "bottle", "seed": "5", "fallback_budget": "0.25", "risk_auroc": "0.8", "random_auroc": "0.6", "risk_delta": "0.2", "fallback_rate": "0.25", "risk_count": "10", "random_count": "10", "total": "40"},
        {"dataset": "MVTec", "category": "cable", "seed": "5", "fallback_budget": "0.25", "risk_auroc": "0.7", "random_auroc": "0.6", "risk_delta": "0.1", "fallback_rate": "0.25", "risk_count": "10", "random_count": "10", "total": "40"},
    ]
    if duplicate:
        raw[1]["category"] = "bottle"
    _write_csv(directory / "raw_results.csv", list(raw[0]), raw)
    _write_csv(directory / "routing_results.csv", ["matched_counts"], [{"matched_counts": "True"}, {"matched_counts": "True"}])
    _write_csv(directory / "main_results.csv", ["dataset", "unit_count", "mean_risk_minus_random_auroc"], [{"dataset": "MVTec", "unit_count": "2", "mean_risk_minus_random_auroc": "0.15"}])
    _write_csv(directory / "stats_results.csv", ["dataset", "mean_delta", "ci95_low", "ci95_high"], [{"dataset": "MVTec", "mean_delta": "0.15", "ci95_low": "0.1", "ci95_high": "0.2"}])
    _write_csv(directory / "ablation_results.csv", ["status", "reason"], [{"status": "NOT_AVAILABLE", "reason": "not sourced"}])
    _write_csv(directory / "efficiency_results.csv", ["path", "batch_size", "cuda_synchronize", "comparison_scope"], [
        {"path": "fast", "batch_size": "1", "cuda_synchronize": "True", "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows"},
        {"path": "full", "batch_size": "1", "cuda_synchronize": "True", "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows"},
        {"path": "risk", "batch_size": "1", "cuda_synchronize": "True", "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows"},
    ])
    (directory / "manifest.json").write_text(json.dumps({"source_row_count": 2}), encoding="utf-8")
    return directory


def test_audit_passes_integrity_checks_and_warns_for_missing_ablation(tmp_path):
    report = audit_canonical_results(_canonical_directory(tmp_path), expected_units={"MVTec": 2})

    assert report["summary"]["fail"] == 0
    assert report["summary"]["warn"] == 1
    assert report["status"] == "WARN"
    assert any(check["name"] == "matched_fallback_counts" and check["status"] == "PASS" for check in report["checks"])


def test_audit_fails_for_duplicate_category_seed_rows(tmp_path):
    report = audit_canonical_results(_canonical_directory(tmp_path, duplicate=True), expected_units={"MVTec": 2})

    assert report["status"] == "FAIL"
    assert any(check["name"] == "unique_category_seed_rows" and check["status"] == "FAIL" for check in report["checks"])


def test_audit_reports_missing_fallback_provenance_as_failure(tmp_path):
    directory = _canonical_directory(tmp_path)
    rows = list(csv.DictReader((directory / "raw_results.csv").open(newline="", encoding="utf-8")))
    rows[0]["risk_count"] = ""
    _write_csv(directory / "raw_results.csv", list(rows[0]), rows)

    report = audit_canonical_results(directory, expected_units={"MVTec": 2})

    assert report["status"] == "FAIL"
    assert any(check["name"] == "matched_fallback_counts" and check["status"] == "FAIL" for check in report["checks"])


def test_audit_accepts_the_current_strong_routing_canonical_schema():
    report = audit_canonical_results(PROJECT_ROOT / "05_运行记录" / "canonical_v3")

    assert report["status"] == "PASS"
    assert report["summary"]["fail"] == 0
    assert any(check["name"] == "matched_exact_quota" and check["status"] == "PASS" for check in report["checks"])


def test_audit_accepts_the_current_budget_sensitivity_canonical_schema():
    report = audit_canonical_results(PROJECT_ROOT / "05_运行记录" / "canonical_v3_budget")

    assert report["status"] == "PASS"
    assert report["summary"]["fail"] == 0
    assert any(check["name"] == "budget_row_counts" and check["status"] == "PASS" for check in report["checks"])


def test_strong_routing_audit_rejects_extra_main_summary_row(tmp_path):
    source = PROJECT_ROOT / "05_运行记录" / "canonical_v3"
    directory = tmp_path / "canonical_v3"
    directory.mkdir()
    for name in ("raw_results.csv", "main_results.csv", "manifest.json"):
        (directory / name).write_bytes((source / name).read_bytes())
    rows = list(csv.DictReader((directory / "main_results.csv").open(newline="", encoding="utf-8")))
    rows.append(rows[0].copy())
    _write_csv(directory / "main_results.csv", list(rows[0]), rows)

    report = audit_canonical_results(directory)

    assert report["status"] == "FAIL"
    assert any(check["name"] == "main_table_aggregation" and check["status"] == "FAIL" for check in report["checks"])


def test_strong_routing_audit_requires_fixed_budget_and_all_three_seeds(tmp_path):
    source = PROJECT_ROOT / "05_运行记录" / "canonical_v3"
    directory = tmp_path / "canonical_v3"
    directory.mkdir()
    for name in ("raw_results.csv", "main_results.csv", "manifest.json"):
        (directory / name).write_bytes((source / name).read_bytes())
    rows = list(csv.DictReader((directory / "raw_results.csv").open(newline="", encoding="utf-8")))
    rows[0]["fallback_budget"] = "0.10"
    rows[1]["seed"] = "5"
    _write_csv(directory / "raw_results.csv", list(rows[0]), rows)

    report = audit_canonical_results(directory)

    assert report["status"] == "FAIL"
    assert any(check["name"] == "fixed_budget_and_seed_layout" and check["status"] == "FAIL" for check in report["checks"])


def test_budget_audit_rejects_extra_summary_row_and_requires_mvtec_layout(tmp_path):
    source = PROJECT_ROOT / "05_运行记录" / "canonical_v3_budget"
    directory = tmp_path / "canonical_v3_budget"
    directory.mkdir()
    for name in ("raw_results.csv", "budget_sensitivity.csv", "manifest.json"):
        (directory / name).write_bytes((source / name).read_bytes())
    summary_rows = list(csv.DictReader((directory / "budget_sensitivity.csv").open(newline="", encoding="utf-8")))
    summary_rows.append(summary_rows[0].copy())
    _write_csv(directory / "budget_sensitivity.csv", list(summary_rows[0]), summary_rows)
    raw_rows = list(csv.DictReader((directory / "raw_results.csv").open(newline="", encoding="utf-8")))
    raw_rows[0]["dataset"] = "MPDD"
    raw_rows[1]["category"] = raw_rows[0]["category"]
    _write_csv(directory / "raw_results.csv", list(raw_rows[0]), raw_rows)

    report = audit_canonical_results(directory)

    assert report["status"] == "FAIL"
    assert any(check["name"] == "budget_summary_aggregation" and check["status"] == "FAIL" for check in report["checks"])
    assert any(check["name"] == "budget_dataset_category_seed_layout" and check["status"] == "FAIL" for check in report["checks"])
