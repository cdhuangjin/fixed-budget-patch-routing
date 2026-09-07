import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_mvtec_padim import padim_scores


class PadimBaselineTests(unittest.TestCase):
    def test_scores_are_finite_and_samplewise(self):
        features = torch.zeros(3, 4, 2)
        mean = torch.zeros(4, 2)
        variance = torch.ones(4, 2)
        scores = padim_scores(features, mean, variance, "cpu")
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(torch.isfinite(torch.from_numpy(scores)).all())


if __name__ == "__main__":
    unittest.main()
