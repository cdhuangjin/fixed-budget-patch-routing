import csv
import json
from pathlib import Path

from materialize_canonical_v2 import materialize


def _source() -> dict:
    return {
        "status": "canonical_strict_quota_main_table",
        "protocol": {
            "fallback_budget": 0.25,
            "route": "strict_quota",
            "scope": "test-free global score ranking; not an online-prefix guarantee",
        },
        "rows": [
            {
                "dataset": "MVTec",
                "category": "bottle",
                "seed": 5,
                "fallback_budget": 0.25,
                "fast_only_auroc": 0.70,
                "full_only_auroc": 0.90,
                "risk_auroc": 0.80,
                "random_auroc": 0.60,
                "risk_delta": 0.20,
                "fallback_rate": 0.25,
                "risk_count": 10,
                "random_count": 10,
                "total": 40,
                "route_source": "test_free_exact_quota_score_ranking",
            },
            {
                "dataset": "MPDD",
                "category": "widget",
                "seed": 5,
                "fallback_budget": 0.25,
                "fast_only_auroc": 0.50,
                "risk_auroc": 0.60,
                "random_auroc": 0.55,
                "risk_delta": 0.05,
                "fallback_rate": 0.25,
                "risk_count": 5,
                "random_count": 5,
                "total": 20,
                "route_source": "test_free_exact_quota_score_ranking",
            },
        ],
    }


def test_materialize_creates_traceable_strict_quota_tables(tmp_path):
    source_path = tmp_path / "strict_quota.json"
    source = _source()
    source["rows"][0].update({"fast_mean_ms": 2.0, "full_mean_ms": 4.0, "risk_mean_ms": 5.0})
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output_dir = tmp_path / "canonical_v2"

    manifest = materialize(source_path, output_dir, bootstrap_reps=100, bootstrap_seed=5)

    assert manifest["schema_version"] == "canonical_v2_strict_quota_v1"
    assert manifest["source_row_count"] == 2
    assert manifest["protocol"]["route"] == "strict_quota"

    with (output_dir / "routing_results.csv").open(newline="", encoding="utf-8") as handle:
        routing = list(csv.DictReader(handle))
    assert routing[0]["risk_minus_random_auroc"] == "0.2"
    assert routing[1]["matched_counts"] == "True"

    with (output_dir / "main_results.csv").open(newline="", encoding="utf-8") as handle:
        main = {row["dataset"]: row for row in csv.DictReader(handle)}
    assert main["MVTec"]["mean_full_only_auroc"] == "0.9"
    assert main["MPDD"]["mean_full_only_auroc"] == ""

    with (output_dir / "stats_results.csv").open(newline="", encoding="utf-8") as handle:
        stats = {row["dataset"]: row for row in csv.DictReader(handle)}
    assert stats["MVTec"]["ci_method"] == "paired_bootstrap"
    assert stats["MVTec"]["bootstrap_reps"] == "100"
    assert stats["MVTec"]["bootstrap_seed"] == "5"

    with (output_dir / "ablation_results.csv").open(newline="", encoding="utf-8") as handle:
        ablation = list(csv.DictReader(handle))
    assert ablation == [{"status": "NOT_AVAILABLE", "reason": "No strict-quota ablation source was materialized."}]


def test_materialize_keeps_audited_latency_as_separate_system_evidence(tmp_path):
    source_path = tmp_path / "strict_quota.json"
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    latency_path = tmp_path / "latency.json"
    latency_path.write_text(
        json.dumps(
            {
                "n_images": 40,
                "fallback_rate_mean": 0.25,
                "fallback_rate_by_category": [0.25, 0.25],
                "latency_protocol": {
                    "batch_size": 1,
                    "warmup_per_cached_image": 1,
                    "repeats_per_cached_image": 20,
                    "cuda_synchronize": True,
                    "memory_bank_images_per_category": "formal_fit_split",
                },
                "latency_ms": {
                    "fast": {"n": 40, "mean_ms": 2.0, "p50_ms": 1.9, "p95_ms": 2.4},
                    "full": {"n": 40, "mean_ms": 3.0, "p50_ms": 2.9, "p95_ms": 3.4},
                    "risk": {"n": 40, "mean_ms": 5.0, "p50_ms": 4.9, "p95_ms": 5.4},
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "canonical_v2"

    manifest = materialize(
        source_path,
        output_dir,
        latency_source_path=latency_path,
        bootstrap_reps=100,
        bootstrap_seed=5,
    )

    assert manifest["latency_audit"]["source_file"] == str(latency_path)
    assert manifest["latency_audit"]["protocol"]["batch_size"] == 1
    assert manifest["latency_audit"]["is_separate_from_accuracy_results"] is True
    with (output_dir / "efficiency_results.csv").open(newline="", encoding="utf-8") as handle:
        efficiency = {row["path"]: row for row in csv.DictReader(handle)}
    assert efficiency["risk"]["mean_ms"] == "5.0"
    assert efficiency["risk"]["p95_ms"] == "5.4"
    assert efficiency["risk"]["comparison_scope"] == "separate_system_audit_not_paired_with_accuracy_rows"


def test_materialize_rejects_non_strict_latency_audit(tmp_path):
    source_path = tmp_path / "strict_quota.json"
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    latency_path = tmp_path / "invalid_latency.json"
    latency_path.write_text(
        json.dumps(
            {
                "n_images": 40,
                "fallback_rate_mean": 0.30,
                "fallback_rate_by_category": [0.30, 0.30],
                "latency_protocol": {"batch_size": 1, "cuda_synchronize": True},
                "latency_ms": {"fast": {}, "full": {}, "risk": {}},
            }
        ),
        encoding="utf-8",
    )

    try:
        materialize(source_path, tmp_path / "canonical_v2", latency_source_path=latency_path)
    except ValueError as error:
        assert "25% strict-quota" in str(error)
    else:
        raise AssertionError("non-strict latency audit should be rejected")
