import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_mvtec_selective import evaluate_records


class TinyModel(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = float(bias)

    def forward(self, images):
        batch = images.shape[0]
        mean = images.mean(dim=(1, 2, 3)) + self.bias
        logits = torch.stack([1.0 - mean, mean], dim=1)
        route = {
            "pooled_feature": images.mean(dim=(2, 3)),
            "uncertainty": torch.sigmoid(mean),
        }
        return logits, route


class MVTecSelectiveEvaluatorTests(unittest.TestCase):
    def test_cpu_one_batch_emits_required_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = []
            for index, value in enumerate((32, 48, 220, 240)):
                path = root / f"{index}.png"
                Image.new("RGB", (12, 12), color=(value, value, value)).save(path)
                records.append({"image_path": str(path), "is_anomaly": int(index >= 2)})
            result = evaluate_records(
                records[:2], records[:2], records[2:], TinyModel(0.0), TinyModel(0.1),
                device="cpu", batch_size=2, fallback_budget=0.5,
            )
            for mode in ("full_only", "sparse_only", "random_fallback", "risk_fallback"):
                self.assertIn("image_auroc", result[mode])
                self.assertIn("recall_at_fpr", result[mode])
                self.assertIn("fallback_rate", result[mode])
                self.assertIn("p95_latency_ms", result[mode])
            self.assertEqual(result["threshold_source"], "validation_normal_only")


if __name__ == "__main__":
    unittest.main()
