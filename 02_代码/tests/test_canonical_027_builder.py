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
                "full_only": {"image_auroc": 0.8, "mean_ms_estimated": 4.0},
                "risk_fallback": {"image_auroc": 0.9, "fallback_rate": 0.2, "mean_ms_estimated": 6.0},
                "random_fallback": {"image_auroc": 0.7},
            }
        ),
        encoding="utf-8",
    )

    rows = collect_mvtec(tmp_path)

    assert rows[0]["fast_mean_ms"] == 4.0
    assert rows[0]["risk_mean_ms"] == 6.0


def test_collect_mvtec_uses_matched_strict_quota_route_when_requested(tmp_path):
    seed_dir = tmp_path / "seed5"
    seed_dir.mkdir()
    (seed_dir / "bottle.json").write_text(
        json.dumps(
            {
                "category": "bottle",
                "fallback_budget": 0.25,
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
    assert row["fallback_rate"] == 0.25
