import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from analysis_protocol import paired_bootstrap_delta_ci, select_common_p95_budgets, summarize_category_metrics


class SelectiveStatisticsTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_paired(self):
        first = paired_bootstrap_delta_ci([0.8, 0.7, 0.9], [0.6, 0.65, 0.75], reps=200, seed=17)
        second = paired_bootstrap_delta_ci([0.8, 0.7, 0.9], [0.6, 0.65, 0.75], reps=200, seed=17)
        self.assertEqual(first, second)
        self.assertGreater(first["mean"], 0.0)

    def test_budget_alignment_uses_validation_latency_only(self):
        selected = select_common_p95_budgets(
            {"risk": [{"budget": 0.1, "p95_ms": 4.0}, {"budget": 0.3, "p95_ms": 6.0}],
             "random": [{"budget": 0.1, "p95_ms": 3.0}, {"budget": 0.3, "p95_ms": 7.0}]},
            target_p95_ms=6.0,
        )
        self.assertEqual(selected["risk"]["budget"], 0.3)
        self.assertEqual(selected["random"]["budget"], 0.1)

    def test_category_summary_contains_ci(self):
        rows = [{"risk": {"image_auroc": value, "recall_at_fpr": value, "fallback_rate": 0.2, "mean_ms": 1.0, "p95_latency_ms": 2.0}}
                for value in (0.7, 0.8, 0.9)]
        summary = summarize_category_metrics(rows, reps=100, seed=17)
        self.assertEqual(summary["risk"]["image_auroc"]["reps"], 100)
        self.assertAlmostEqual(summary["risk"]["image_auroc"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
