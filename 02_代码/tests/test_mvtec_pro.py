import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_mvtec_pro import connected_components, pro_auc, pro_curve


class MvtecProTests(unittest.TestCase):
    def test_connected_components_finds_two_regions(self):
        mask = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=bool)
        self.assertEqual(len(connected_components(mask)), 2)

    def test_perfect_score_map_has_unit_pro(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        scores = mask.astype(float)
        curve = pro_curve([scores], [mask], thresholds=[1.0, 0.5, 0.0])
        self.assertGreaterEqual(curve[:, 1].max(), 1.0)
        self.assertAlmostEqual(pro_auc([scores], [mask], max_fpr=0.3, thresholds=[1.0, 0.5, 0.0]), 1.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            pro_curve([np.zeros((2, 2))], [np.zeros((3, 3), dtype=bool)])


if __name__ == "__main__":
    unittest.main()
