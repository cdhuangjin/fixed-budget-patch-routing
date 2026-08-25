import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_cifar100c import load_cifar100c, select_p95_budget


class CIFAR100CTests(unittest.TestCase):
    def test_loader_reads_one_corruption_severity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "gaussian_noise.npy", np.zeros((30000, 32, 32, 3), dtype=np.uint8))
            np.save(root / "labels.npy", np.arange(10000) % 100)
            images, labels = load_cifar100c(root, "gaussian_noise", severity=2)
            self.assertEqual(images.shape, (10000, 3, 32, 32))
            self.assertEqual(labels.shape, (10000,))

    def test_budget_selects_closest_latency_without_using_test_labels(self):
        points = [{"budget": 32, "p95_ms": 1.2}, {"budget": 64, "p95_ms": 1.8}]
        selected = select_p95_budget(points, target_p95_ms=1.7)
        self.assertEqual(selected["budget"], 32)

    def test_budget_selection_rejects_unmet_fixed_latency_target(self):
        points = [{"budget": 32, "p95_ms": 2.2}, {"budget": 64, "p95_ms": 2.8}]
        with self.assertRaises(ValueError):
            select_p95_budget(points, target_p95_ms=2.0)


if __name__ == "__main__":
    unittest.main()
