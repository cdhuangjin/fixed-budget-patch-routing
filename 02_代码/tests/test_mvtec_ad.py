import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from mvtec_ad import MVTecADIndex, split_good_images


class MVTecADTests(unittest.TestCase):
    def _make_dataset(self, root):
        for relative in (
            "bottle/train/good/001.png",
            "bottle/train/good/002.png",
            "bottle/train/good/003.png",
            "bottle/test/good/004.png",
            "bottle/test/broken/005.png",
            "bottle/ground_truth/broken/005_mask.png",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    def test_index_reads_good_train_and_test_anomaly_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            dataset = MVTecADIndex(root)
            self.assertEqual(dataset.categories(), ["bottle"])
            self.assertEqual(dataset.split_counts("bottle", "train"), {"good": 3})
            self.assertEqual(dataset.split_counts("bottle", "test"), {"good": 1, "broken": 1})
            anomaly = next(record for record in dataset.samples("bottle", "test") if record["is_anomaly"])
            self.assertTrue(anomaly["is_anomaly"])
            self.assertTrue(anomaly["mask_path"].endswith("005_mask.png"))

    def test_validation_split_uses_only_good_training_images(self):
        paths = [Path(f"{index}.png") for index in range(4)]
        train, validation = split_good_images(paths, validation_fraction=0.5, seed=17)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertEqual(len(train) + len(validation), len(paths))

    def test_missing_category_structure_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                MVTecADIndex(Path(temp_dir)).samples("bottle", "test")


if __name__ == "__main__":
    unittest.main()
