import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from analysis_protocol import summarize_latency


class LatencySummaryTests(unittest.TestCase):
    def test_summary_has_reproducible_percentiles(self):
        result = summarize_latency([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["mean_ms"], 3.0)
        self.assertIn("p95_ms", result)
        self.assertIn("iqr_ms", result)

    def test_empty_latency_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_latency([])


if __name__ == "__main__":
    unittest.main()
