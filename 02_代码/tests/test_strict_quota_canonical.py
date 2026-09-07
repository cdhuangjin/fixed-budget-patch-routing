import json

import pytest

from build_strict_quota_027_canonical import collect_external_strict_quota


def _item(*, matched=True):
    return {
        "category": "widget",
        "fast_only_auroc": 0.7,
        "risk_combined_auroc": 0.9,
        "random_combined_auroc": 0.8,
        "risk_delta": 0.1,
        "budget": 0.25,
        "seed": 5,
        "fallback_rate": 0.25,
        "risk_count": 10,
        "total": 40,
        "route": {
            "route_name": "strict_quota",
            "target_fallback_count": 10,
            "actual_fallback_count": 10,
            "actual_rate": 0.25,
            "route_source": "test_free_exact_quota_score_ranking",
        },
        "random_route": {
            "route_name": "random_matched",
            "actual_fallback_count": 10 if matched else 9,
            "actual_rate": 0.25 if matched else 0.225,
        },
    }


def test_collect_external_strict_quota_records_matched_routes(tmp_path):
    output = tmp_path / "visa" / "seed5"
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps([_item()]), encoding="utf-8")

    rows = collect_external_strict_quota(tmp_path)

    assert rows == [
        {
            "dataset": "VisA",
            "category": "widget",
            "seed": 5,
            "fallback_budget": 0.25,
            "fast_only_auroc": 0.7,
            "risk_auroc": 0.9,
            "random_auroc": 0.8,
            "risk_delta": 0.1,
            "fallback_rate": 0.25,
            "risk_count": 10,
            "random_count": 10,
            "total": 40,
            "route_source": "test_free_exact_quota_score_ranking",
        }
    ]


def test_collect_external_strict_quota_rejects_unmatched_random_count(tmp_path):
    output = tmp_path / "mpdd" / "seed5"
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps([_item(matched=False)]), encoding="utf-8")

    with pytest.raises(ValueError, match="matched random quota"):
        collect_external_strict_quota(tmp_path)
