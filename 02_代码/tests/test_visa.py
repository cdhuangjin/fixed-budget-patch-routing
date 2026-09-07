import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from visa import VisAIndex


class VisATests(unittest.TestCase):
    def _make_dataset(self, root):
        for relative in (
            "VisA_20220922/candle/Data/Images/Normal/0000.JPG",
            "VisA_20220922/candle/Data/Images/Normal/0001.JPG",
            "VisA_20220922/candle/Data/Images/Anomaly/000.JPG",
            "VisA_20220922/candle/Data/Masks/Anomaly/000.png",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake")

    def test_index_reads_visA_layout_and_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            dataset = VisAIndex(root)
            self.assertEqual(dataset.categories(), ["candle"])
            self.assertEqual(dataset.split_counts("candle", "train"), {"good": 2})
            self.assertEqual(dataset.split_counts("candle", "test"), {"good": 2, "anomaly": 1})
            anomaly = next(record for record in dataset.samples("candle", "test") if record["is_anomaly"])
            self.assertTrue(anomaly["mask_path"].endswith("000.png"))

    def test_missing_category_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                VisAIndex(temp_dir).samples("candle", "test")


if __name__ == "__main__":
    unittest.main()
