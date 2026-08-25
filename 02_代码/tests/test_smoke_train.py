import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from smoke_train import build_model, make_synthetic_loader, train_steps


class SmokeTrainingTests(unittest.TestCase):
    def test_synthetic_loader_has_cifar100_shapes(self):
        loader = make_synthetic_loader(samples=4, batch_size=2, image_size=32, num_classes=100)
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertTrue(torch.all((labels >= 0) & (labels < 100)))

    def test_all_methods_produce_logits_and_training_history(self):
        for method in ("full", "fixed_sparse", "random_sparse", "difficulty_only", "uncertainty_only", "rata"):
            model = build_model(method, image_size=32, num_classes=100)
            loader = make_synthetic_loader(samples=4, batch_size=2, image_size=32, num_classes=100)
            with tempfile.TemporaryDirectory() as temp_dir:
                result = train_steps(model, loader, steps=2, output_dir=temp_dir)
                self.assertEqual(result["steps_completed"], 2)
                self.assertEqual(len(result["loss"]), 2)
                self.assertTrue(Path(temp_dir, "checkpoint.pt").exists())

    def test_rata_budget_is_below_fixed_sparse_maximum(self):
        fixed = build_model("fixed_sparse", image_size=32, num_classes=100)
        rata = build_model("rata", image_size=32, num_classes=100)
        self.assertEqual(fixed.config.k_base, 32)
        self.assertEqual(rata.config.k_max, fixed.config.k_base)
        self.assertLess(rata.config.k_base, rata.config.k_max)


if __name__ == "__main__":
    unittest.main()
