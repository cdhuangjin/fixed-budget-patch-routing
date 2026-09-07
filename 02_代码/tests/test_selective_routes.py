import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from selective_routes import (
    oracle_matched_route,
    random_matched_route,
    risk_route,
    score_matched_route,
    quota_route,
    online_prefix_quota_route,
    threshold_for_budget,
    conformal_threshold,
    strict_quota_combined_scores,
)


class SelectiveRouteTests(unittest.TestCase):
    def test_risk_route_is_threshold_only(self):
        decision = risk_route([0.1, 0.8, 0.4], 0.5)
        np.testing.assert_array_equal(decision.mask, [False, True, False])
        self.assertEqual(decision.actual_fallback_count, 1)
        self.assertEqual(decision.route_source, "validation_calibrated_score")

    def test_random_route_matches_exact_count_and_seed(self):
        first = random_matched_route(10, 3, seed=17)
        second = random_matched_route(10, 3, seed=17)
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertEqual(first.actual_fallback_count, 3)

    def test_score_route_matches_exact_count_and_is_stable_on_ties(self):
        decision = score_matched_route([0.5, 0.9, 0.9, 0.1], 2)
        np.testing.assert_array_equal(decision.mask, [False, True, True, False])

    def test_oracle_is_explicitly_upper_bound(self):
        decision = oracle_matched_route([0, 1, 0, 1], 2)
        np.testing.assert_array_equal(decision.mask, [False, True, False, True])
        self.assertEqual(decision.route_source, "test_labels_upper_bound_only")

    def test_budget_threshold_is_reproducible(self):
        self.assertAlmostEqual(threshold_for_budget([0.1, 0.2, 0.8, 0.9], 0.5), 0.5)

    def test_quota_route_enforces_declared_budget(self):
        decision = quota_route([0.1, 0.8, 0.4, 0.9], 0.5)
        np.testing.assert_array_equal(decision.mask, [False, True, False, True])
        self.assertEqual(decision.actual_fallback_count, 2)
        self.assertEqual(decision.route_source, "test_free_exact_quota_score_ranking")

    def test_strict_quota_combined_scores_match_random_control_budget(self):
        risk_scores, random_scores, risk, control = strict_quota_combined_scores(
            [0.1, 0.8, 0.4, 0.9], fallback_budget=0.5, seed=17, boost=1.5
        )
        self.assertEqual(risk.actual_fallback_count, 2)
        self.assertEqual(control.actual_fallback_count, 2)
        np.testing.assert_allclose(risk_scores, [0.1, 1.2, 0.4, 1.35])
        self.assertEqual(np.count_nonzero(random_scores != [0.1, 0.8, 0.4, 0.9]), 2)

    def test_conformal_threshold_is_upper_calibration_quantile(self):
        self.assertEqual(conformal_threshold([0.1, 0.2, 0.8, 0.9], 0.5), 0.8)

    def test_online_prefix_quota_uses_only_current_batches_and_exact_total(self):
        decision = online_prefix_quota_route([0.1, 0.9, 0.2, 0.8, 0.3], 0.4, batch_size=2)
        np.testing.assert_array_equal(decision.mask, [False, True, False, True, False])
        self.assertEqual(decision.actual_fallback_count, 2)
        self.assertEqual(decision.route_source, "online_batch_prefix_exact_quota_score_ranking")

    def test_invalid_count_is_rejected(self):
        with self.assertRaises(ValueError):
            random_matched_route(3, 4)


if __name__ == "__main__":
    unittest.main()
