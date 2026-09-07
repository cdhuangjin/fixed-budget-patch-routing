import csv

from make_canonical_v2_figures import generate_figures


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_figures_uses_only_canonical_v2_tables(tmp_path):
    canonical_dir = tmp_path / "canonical_v2"
    canonical_dir.mkdir()
    _write_csv(
        canonical_dir / "stats_results.csv",
        ["dataset", "mean_delta", "ci95_low", "ci95_high", "unit_count"],
        [
            {"dataset": "MVTec", "mean_delta": "0.09", "ci95_low": "0.07", "ci95_high": "0.11", "unit_count": "75"},
            {"dataset": "MPDD", "mean_delta": "0.04", "ci95_low": "0.02", "ci95_high": "0.06", "unit_count": "30"},
        ],
    )
    _write_csv(
        canonical_dir / "efficiency_results.csv",
        ["dataset", "path", "mean_ms", "p50_ms", "p95_ms", "n_images", "fallback_rate_mean", "batch_size", "repeats_per_cached_image", "cuda_synchronize", "comparison_scope"],
        [
            {"dataset": "MVTec", "path": "fast", "mean_ms": "2", "p50_ms": "1.9", "p95_ms": "2.4", "n_images": "40", "fallback_rate_mean": "0.25", "batch_size": "1", "repeats_per_cached_image": "20", "cuda_synchronize": "True", "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows"},
            {"dataset": "MVTec", "path": "full", "mean_ms": "3", "p50_ms": "2.9", "p95_ms": "3.4", "n_images": "40", "fallback_rate_mean": "0.25", "batch_size": "1", "repeats_per_cached_image": "20", "cuda_synchronize": "True", "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows"},
            {"dataset": "MVTec", "path": "risk", "mean_ms": "5", "p50_ms": "4.9", "p95_ms": "5.4", "n_images": "40", "fallback_rate_mean": "0.25", "batch_size": "1", "repeats_per_cached_image": "20", "cuda_synchronize": "True", "comparison_scope": "separate_system_audit_not_paired_with_accuracy_rows"},
        ],
    )
    output_dir = tmp_path / "figures"

    outputs = generate_figures(canonical_dir, output_dir)

    assert set(outputs) == {"allocation_effect", "latency_audit"}
    for base_path in outputs.values():
        for suffix in (".pdf", ".svg", ".tiff", ".png"):
            assert base_path.with_suffix(suffix).is_file()


def test_generate_figures_rejects_non_audited_efficiency_table(tmp_path):
    canonical_dir = tmp_path / "canonical_v2"
    canonical_dir.mkdir()
    _write_csv(canonical_dir / "stats_results.csv", ["dataset", "mean_delta", "ci95_low", "ci95_high", "unit_count"], [{"dataset": "MVTec", "mean_delta": "0.09", "ci95_low": "0.07", "ci95_high": "0.11", "unit_count": "75"}])
    _write_csv(canonical_dir / "efficiency_results.csv", ["status", "reason"], [{"status": "NOT_AVAILABLE", "reason": "missing"}])

    try:
        generate_figures(canonical_dir, tmp_path / "figures")
    except ValueError as error:
        assert "audited efficiency" in str(error)
    else:
        raise AssertionError("placeholder efficiency table should be rejected")
