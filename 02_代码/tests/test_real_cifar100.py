import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from real_cifar100 import (
    compute_class_metrics,
    compute_ece,
    evaluate_model,
    hard_example_indices,
    make_loader,
    seed_everything,
    summarize_hard_examples,
)
from smoke_train import build_model


class RealCIFAR100Tests(unittest.TestCase):
    def test_hard_examples_and_class_metrics_are_deterministic(self):
        losses = [0.1, 0.9, 0.8, 0.2]
        correct = [True, False, True, False]
        hard = summarize_hard_examples(losses, correct, fraction=0.5)
        self.assertEqual(hard["count"], 2)
        self.assertEqual(hard["accuracy"], 0.5)
        metrics = compute_class_metrics([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_fixed_hard_indices_can_be_shared_across_methods(self):
        reference_losses = [0.1, 0.9, 0.8, 0.2]
        indices = hard_example_indices(reference_losses, fraction=0.5)
        self.assertEqual(indices, [1, 2])
        method_a = summarize_hard_examples([0.0] * 4, [True, False, True, False], hard_indices=indices)
        method_b = summarize_hard_examples([0.0] * 4, [False, True, False, True], hard_indices=indices)
        self.assertEqual(method_a["count"], method_b["count"])
        self.assertAlmostEqual(method_a["accuracy"], 0.5)
        self.assertAlmostEqual(method_b["accuracy"], 0.5)

    def test_ece_is_zero_for_perfectly_calibrated_single_bin(self):
        ece = compute_ece([1.0, 1.0], [True, True], bins=10)
        self.assertAlmostEqual(ece, 0.0)

    def test_seed_everything_reproduces_model_initialization(self):
        seed_everything(17)
        first = build_model("full", image_size=32, num_classes=100)
        seed_everything(17)
        second = build_model("full", image_size=32, num_classes=100)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))

    def test_make_loader_preserves_image_and_label_shapes(self):
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        loader = make_loader(TensorDataset(images, labels), batch_size=2, shuffle=False)
        batch_images, batch_labels = next(iter(loader))
        self.assertEqual(tuple(batch_images.shape), (2, 3, 32, 32))
        self.assertEqual(tuple(batch_labels.shape), (2,))

    def test_evaluate_model_reports_accuracy_and_hard_example_metrics(self):
        model = build_model("full", image_size=32, num_classes=100)
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        metrics = evaluate_model(
            model,
            DataLoader(TensorDataset(images, labels), batch_size=2),
            "cpu",
            hard_indices=[0, 1],
        )
        self.assertIn("accuracy", metrics)
        self.assertIn("worst_10pct_accuracy", metrics)
        self.assertEqual(metrics["hard_example_count"], 2)
        self.assertIn("p95_latency_ms", metrics)
        self.assertEqual(metrics["samples"], 4)


if __name__ == "__main__":
    unittest.main()
