import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from analysis_protocol import bootstrap_mean_ci, select_common_p95_budgets


class AnalysisProtocolTests(unittest.TestCase):
    def test_common_p95_selection_uses_validation_only(self):
        points = {
            "full": [{"budget": 32, "p95_ms": 1.0}, {"budget": 64, "p95_ms": 2.0}],
            "rata": [{"budget": 32, "p95_ms": 1.2}, {"budget": 64, "p95_ms": 2.4}],
        }
        selected = select_common_p95_budgets(points, target_p95_ms=2.1)
        self.assertEqual(selected["full"]["budget"], 64)
        self.assertEqual(selected["rata"]["budget"], 32)

    def test_bootstrap_mean_ci_is_reproducible_and_contains_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        first = bootstrap_mean_ci(values, reps=1000, seed=17)
        second = bootstrap_mean_ci(values, reps=1000, seed=17)
        self.assertEqual(first, second)
        self.assertLessEqual(first["low"], first["mean"])
        self.assertGreaterEqual(first["high"], first["mean"])


if __name__ == "__main__":
    unittest.main()
