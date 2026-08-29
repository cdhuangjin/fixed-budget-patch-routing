import json

from build_canonical_027_main_table import collect_mvtec


def test_collect_mvtec_accepts_v2_estimated_latency_schema(tmp_path):
    seed_dir = tmp_path / "seed41"
    seed_dir.mkdir()
    (seed_dir / "bottle.json").write_text(
        json.dumps(
            {
                "category": "bottle",
                "fallback_budget": 0.25,
                "fast_only": {"image_auroc": 0.7, "mean_ms_estimated": 2.0},
                "full_only": {"image_auroc": 0.8, "mean_ms_estimated": 4.0},
                "risk_fallback": {"image_auroc": 0.9, "fallback_rate": 0.2, "mean_ms_estimated": 6.0},
                "random_fallback": {"image_auroc": 0.7},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_mvtec(tmp_path)

    assert rows[0]["fast_only_auroc"] == 0.7
    assert rows[0]["fast_mean_ms"] == 2.0
    assert rows[0]["risk_mean_ms"] == 6.0


def test_collect_mvtec_uses_matched_strict_quota_route_when_requested(tmp_path):
    seed_dir = tmp_path / "seed5"
    seed_dir.mkdir()
    (seed_dir / "bottle.json").write_text(
        json.dumps(
            {
                "category": "bottle",
                "fallback_budget": 0.25,
                "fast_only": {"image_auroc": 0.65, "mean_ms_estimated": 2.0},
                "full_only": {"image_auroc": 0.8, "mean_ms_estimated": 4.0},
                "risk_fallback": {"image_auroc": 0.99, "fallback_rate": 0.8, "mean_ms_estimated": 7.0},
                "random_fallback": {"image_auroc": 0.5},
                "strict_quota": {"image_auroc": 0.7, "fallback_rate": 0.25, "mean_ms_estimated": 5.0},
                "strict_quota_random": {"image_auroc": 0.6},
            }
        ),
        encoding="utf-8",
    )

    row = collect_mvtec(tmp_path, route="strict_quota")[0]

    assert row["risk_auroc"] == 0.7
    assert row["random_auroc"] == 0.6
    assert row["fast_only_auroc"] == 0.65
    assert row["fallback_rate"] == 0.25


def test_collect_mvtec_preserves_strict_quota_provenance_counts(tmp_path):
    seed_dir = tmp_path / "seed5"
    seed_dir.mkdir()
    (seed_dir / "bottle.json").write_text(
        json.dumps(
            {
                "category": "bottle",
                "fallback_budget": 0.25,
                "fast_only": {"image_auroc": 0.65},
                "full_only": {"image_auroc": 0.8},
                "strict_quota": {"image_auroc": 0.7, "fallback_rate": 0.25},
                "strict_quota_random": {"image_auroc": 0.6},
                "routing": {
                    "strict_quota": {
                        "route_name": "strict_quota",
                        "actual_fallback_count": 10,
                        "actual_rate": 0.25,
                        "route_source": "test_free_exact_quota_score_ranking",
                    },
                    "strict_quota_random": {
                        "route_name": "random_matched",
                        "actual_fallback_count": 10,
                        "actual_rate": 0.25,
                        "route_source": "seeded_uniform_sample_without_replacement",
                    },
                },
                "n_test": 40,
            }
        ),
        encoding="utf-8",
    )

    row = collect_mvtec(tmp_path, route="strict_quota")[0]

    assert row["risk_count"] == 10
    assert row["random_count"] == 10
    assert row["total"] == 40
    assert row["route_source"] == "test_free_exact_quota_score_ranking"
