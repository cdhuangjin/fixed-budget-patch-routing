import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from selective_inference import NormalityRiskScorer, choose_threshold, selective_predict


class SelectiveInferenceTests(unittest.TestCase):
    def test_normal_prototype_score_is_zero_for_the_prototype(self):
        scorer = NormalityRiskScorer()
        scorer.fit(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        score = scorer.score(torch.tensor([[0.5, 0.5]]))
        self.assertAlmostEqual(float(score.item()), 0.0, places=5)

    def test_threshold_selection_respects_fallback_budget(self):
        threshold = choose_threshold([0.1, 0.2, 0.8, 0.9], fallback_budget=0.5)
        self.assertGreaterEqual(threshold, 0.2)

    def test_risk_fallback_calls_full_only_for_high_risk_samples(self):
        calls = []

        def full_predict(indices):
            calls.append(list(indices))
            return {index: index + 10 for index in indices}

        result = selective_predict(
            risk_scores=[0.1, 0.8, 0.2, 0.9],
            fast_predictions={0: 0, 1: 1, 2: 2, 3: 3},
            full_predict=full_predict,
            threshold=0.5,
            fast_latency_ms=1.0,
            full_latency_ms=3.0,
        )
        self.assertEqual(calls, [[1, 3]])
        self.assertEqual(result["fallback_count"], 2)
        self.assertEqual(result["predictions"], {0: 0, 1: 11, 2: 2, 3: 13})
        self.assertAlmostEqual(result["fallback_rate"], 0.5)
        self.assertGreater(result["p95_latency_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
