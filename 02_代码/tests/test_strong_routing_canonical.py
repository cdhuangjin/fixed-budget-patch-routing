import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from build_strong_routing_canonical import collect_external, collect_mvtec


class StrongRoutingCanonicalTests(unittest.TestCase):
    def write_mvtec(self, path: Path, count: int = 8):
        route = {
            "actual_fallback_count": count,
            "route_source": "test_free_exact_quota_score_ranking",
        }
        record = {
            "category": "bottle",
            "seed": 5,
            "fallback_budget": 0.25,
            "n_test": 32,
            "fast_only": {"image_auroc": 0.9, "recall_at_fpr": 0.5},
            "risk_fallback": {"image_auroc": 0.8, "recall_at_fpr": 0.6},
            "strict_quota": {**route, "fallback_rate": 0.25},
            "strict_quota_random": {"image_auroc": 0.7, "recall_at_fpr": 0.4, **route},
            "fast_score": {"image_auroc": 0.75, "recall_at_fpr": 0.5, **route},
            "uncertainty_dispersion": {"image_auroc": 0.65, "recall_at_fpr": 0.4, **route},
            "routing": {
                "strict_quota": route,
                "strict_quota_random": route,
                "fast_score": route,
                "uncertainty_dispersion": route,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")

    def test_collect_mvtec_rejects_unmatched_strong_baseline_budget(self):
        path = Path(self.temp_dir.name) / "seed5" / "bottle.json"
        self.write_mvtec(path)
        rows = collect_mvtec(path.parent.parent)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["fallback_count"], 8)
        self.assertAlmostEqual(row["risk_minus_fast_score"], 0.05)
        self.assertAlmostEqual(row["risk_minus_uncertainty"], 0.15)

        record = json.loads(path.read_text(encoding="utf-8"))
        record["routing"]["fast_score"]["actual_fallback_count"] = 7
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ValueError):
            collect_mvtec(path.parent.parent)

    def test_collect_external_records_all_matched_routes(self):
        path = Path(self.temp_dir.name) / "visa" / "seed5" / "results.json"
        route = {
            "actual_fallback_count": 4,
            "route_source": "test-free_score_ranking",
        }
        record = [{
            "category": "candle",
            "seed": 5,
            "budget": 0.25,
            "total": 16,
            "fallback_rate": 0.25,
            "fast_only_auroc": 0.9,
            "fast_only_recall": 0.5,
            "risk_combined_auroc": 0.8,
            "risk_combined_recall": 0.6,
            "random_combined_auroc": 0.7,
            "random_combined_recall": 0.4,
            "fast_score_combined_auroc": 0.75,
            "fast_score_combined_recall": 0.5,
            "uncertainty_combined_auroc": 0.65,
            "uncertainty_combined_recall": 0.4,
            "route": route,
            "random_route": route,
            "fast_score_route": route,
            "uncertainty_route": route,
        }]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        rows = collect_external(path.parents[2])
        self.assertEqual(rows[0]["dataset"], "VisA")
        self.assertEqual(rows[0]["fallback_count"], 4)
        self.assertAlmostEqual(rows[0]["risk_minus_fast_score"], 0.05)

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
