#!/usr/bin/env python3
"""Materialize auditable Canonical V2 CSV tables from strict-quota results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from analysis_protocol import paired_bootstrap_delta_ci


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_source(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "canonical_strict_quota_main_table":
        raise ValueError("source is not a strict-quota canonical table")
    rows = source.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source has no rows")
    for row in rows:
        if not math.isclose(
            float(row["risk_delta"]),
            float(row["risk_auroc"]) - float(row["random_auroc"]),
            abs_tol=1e-12,
        ):
            raise ValueError("risk_delta does not match risk_auroc minus random_auroc")
        if "risk_count" in row and int(row["risk_count"]) != int(row["random_count"]):
            raise ValueError("source contains unmatched fallback counts")
    return source


def _load_latency_audit(latency_source_path: Path) -> dict[str, Any]:
    """Load a batch-one, CUDA-synchronized, strict-quota system audit."""
    audit = json.loads(latency_source_path.read_text(encoding="utf-8"))
    if not math.isclose(float(audit.get("fallback_rate_mean", -1)), 0.25, abs_tol=1e-12):
        raise ValueError("latency audit is not a 25% strict-quota audit")
    fallback_rates = audit.get("fallback_rate_by_category")
    if not isinstance(fallback_rates, list) or not fallback_rates:
        raise ValueError("latency audit has no category-level fallback rates")
    if not all(math.isclose(float(rate), 0.25, abs_tol=1e-12) for rate in fallback_rates):
        raise ValueError("latency audit is not a 25% strict-quota audit")

    protocol = audit.get("latency_protocol")
    if not isinstance(protocol, dict) or protocol.get("batch_size") != 1:
        raise ValueError("latency audit must use batch_size=1")
    if protocol.get("cuda_synchronize") is not True:
        raise ValueError("latency audit must synchronize CUDA")

    latency_ms = audit.get("latency_ms")
    if not isinstance(latency_ms, dict):
        raise ValueError("latency audit has no latency_ms table")
    for path in ("fast", "full", "risk"):
        metrics = latency_ms.get(path)
        if not isinstance(metrics, dict) or any(key not in metrics for key in ("n", "mean_ms", "p50_ms", "p95_ms")):
            raise ValueError(f"latency audit lacks required metrics for {path}")
    return audit


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    aggregates: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        output = dict(zip(group_keys, key, strict=True))
        output.update(
            unit_count=len(items),
            mean_fast_only_auroc=_mean(float(item["fast_only_auroc"]) for item in items),
            mean_risk_auroc=_mean(float(item["risk_auroc"]) for item in items),
            mean_random_auroc=_mean(float(item["random_auroc"]) for item in items),
            mean_risk_minus_random_auroc=_mean(float(item["risk_delta"]) for item in items),
            mean_fallback_rate=_mean(float(item["fallback_rate"]) for item in items),
        )
        full_values = [float(item["full_only_auroc"]) for item in items if "full_only_auroc" in item]
        output["mean_full_only_auroc"] = _mean(full_values) if full_values else ""
        aggregates.append(output)
    return aggregates


def materialize(
    source_path: Path,
    output_dir: Path,
    *,
    latency_source_path: Path | None = None,
    bootstrap_reps: int = 10_000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Write Canonical V2 tables without mixing strict-quota and legacy results."""
    source = _load_source(source_path)
    latency_audit = _load_latency_audit(latency_source_path) if latency_source_path else None
    rows: list[dict[str, Any]] = source["rows"]
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_fields = [
        "dataset", "category", "seed", "fallback_budget", "fast_only_auroc",
        "full_only_auroc", "risk_auroc", "random_auroc", "risk_delta",
        "fallback_rate", "risk_count", "random_count", "total", "route_source",
    ]
    _write_csv(output_dir / "raw_results.csv", raw_fields, rows)

    routing_rows = []
    for row in rows:
        routing_rows.append({
            "dataset": row["dataset"],
            "category": row["category"],
            "seed": row["seed"],
            "fallback_budget": row["fallback_budget"],
            "risk_auroc": row["risk_auroc"],
            "random_auroc": row["random_auroc"],
            "risk_minus_random_auroc": row["risk_delta"],
            "risk_fallback_count": row.get("risk_count", ""),
            "random_fallback_count": row.get("random_count", ""),
            "matched_counts": row.get("risk_count", "") == row.get("random_count", ""),
            "fallback_rate": row["fallback_rate"],
            "route_source": row.get("route_source", ""),
        })
    _write_csv(
        output_dir / "routing_results.csv",
        list(routing_rows[0]),
        routing_rows,
    )

    main_rows = _aggregate(rows, ["dataset"])
    main_fields = [
        "dataset", "unit_count", "mean_fast_only_auroc", "mean_full_only_auroc",
        "mean_risk_auroc", "mean_random_auroc", "mean_risk_minus_random_auroc",
        "mean_fallback_rate",
    ]
    _write_csv(output_dir / "main_results.csv", main_fields, main_rows)

    budget_rows = _aggregate(rows, ["dataset", "fallback_budget"])
    budget_fields = [
        "dataset", "fallback_budget", "unit_count", "mean_fast_only_auroc",
        "mean_full_only_auroc", "mean_risk_auroc", "mean_random_auroc",
        "mean_risk_minus_random_auroc", "mean_fallback_rate",
    ]
    _write_csv(output_dir / "budget_results.csv", budget_fields, budget_rows)

    stats_rows = []
    for aggregate in main_rows:
        items = [row for row in rows if row["dataset"] == aggregate["dataset"]]
        bootstrap = paired_bootstrap_delta_ci(
            [float(row["risk_auroc"]) for row in items],
            [float(row["random_auroc"]) for row in items],
            reps=bootstrap_reps,
            seed=bootstrap_seed,
        )
        stats_rows.append({
            "dataset": aggregate["dataset"],
            "fallback_budget": source["protocol"]["fallback_budget"],
            "comparison": "strict_quota_risk_minus_matched_random",
            "unit": "category_seed",
            "unit_count": len(items),
            "mean_delta": aggregate["mean_risk_minus_random_auroc"],
            "ci_method": "paired_bootstrap",
            "ci95_low": bootstrap["low"],
            "ci95_high": bootstrap["high"],
            "bootstrap_reps": bootstrap["reps"],
            "bootstrap_seed": bootstrap_seed,
            "note": "Paired bootstrap over category-seed rows; category and seed dependence remains.",
        })
    _write_csv(output_dir / "stats_results.csv", list(stats_rows[0]), stats_rows)

    _write_csv(
        output_dir / "ablation_results.csv",
        ["status", "reason"],
        [{"status": "NOT_AVAILABLE", "reason": "No strict-quota ablation source was materialized."}],
    )
    if latency_audit is None:
        _write_csv(
            output_dir / "efficiency_results.csv",
            ["status", "reason"],
            [{
                "status": "NOT_AVAILABLE",
                "reason": "The strict-quota source contains estimated path costs, not the formal CUDA latency audit.",
            }],
        )
    else:
        efficiency_rows = []
        for path in ("fast", "full", "risk"):
            metrics = latency_audit["latency_ms"][path]
            efficiency_rows.append({
                "dataset": "MVTec",
                "path": path,
                "n_images": latency_audit["n_images"],
                "fallback_rate_mean": latency_audit["fallback_rate_mean"],
                "mean_ms": metrics["mean_ms"],
                "p50_ms": metrics["p50_ms"],
                "p95_ms": metrics["p95_ms"],
                "batch_size": latency_audit["latency_protocol"]["batch_size"],
                "repeats_per_cached_image": latency_audit["latency_protocol"].get("repeats_per_cached_image", ""),
                "cuda_synchronize": latency_audit["latency_protocol"]["cuda_synchronize"],
                "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows",
            })
        _write_csv(output_dir / "efficiency_results.csv", list(efficiency_rows[0]), efficiency_rows)

    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "canonical_v2_strict_quota_v1",
        "source_file": str(source_path),
        "source_sha256": source_digest,
        "source_row_count": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "protocol": source["protocol"],
        "materialized_files": [
            "raw_results.csv", "main_results.csv", "routing_results.csv", "budget_results.csv",
            "ablation_results.csv", "efficiency_results.csv", "stats_results.csv",
        ],
        "scope_limitations": [
            "Only the 25% strict-quota, test-free global ranking protocol is materialized.",
            "Statistics are descriptive category-seed summaries and do not treat rows as independent datasets.",
        ],
    }
    if latency_source_path is None:
        manifest["scope_limitations"].insert(
            1, "Ablation and formal latency tables are explicitly unavailable until sourced from audited records."
        )
    else:
        manifest["latency_audit"] = {
            "source_file": str(latency_source_path),
            "source_sha256": hashlib.sha256(latency_source_path.read_bytes()).hexdigest(),
            "dataset": "MVTec",
            "n_images": latency_audit["n_images"],
            "protocol": latency_audit["latency_protocol"],
            "fallback_rate_mean": latency_audit["fallback_rate_mean"],
            "is_separate_from_accuracy_results": True,
            "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows",
        }
        manifest["scope_limitations"].insert(
            1, "Latency is a separate MVTec system audit and is not paired with the five-seed accuracy rows."
        )
        manifest["scope_limitations"].insert(
            2, "No strict-quota ablation source was materialized."
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--latency-source", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    manifest = materialize(
        args.source,
        args.output_dir,
        latency_source_path=args.latency_source,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
