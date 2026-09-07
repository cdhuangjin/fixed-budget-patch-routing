"""VisA directory indexing for normal-only industrial anomaly evaluation."""

import argparse
import random
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def split_good_images(paths, validation_fraction=0.1, seed=17):
    paths = list(paths)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1) if len(shuffled) > 1 else 0
    return sorted(shuffled[validation_count:]), sorted(shuffled[:validation_count])


class VisAIndex:
    """Index the extracted VisA layout.

    Expected layout: <root>/<category>/Data/Images/{Normal,Anomaly}
    and, for anomalous images, <root>/<category>/Data/Masks/Anomaly.
    The archive's normal images are exposed as the training pool and as the
    normal portion of the test split; anomaly images are test-only.
    """

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"VisA root does not exist: {self.root}")
        if (self.root / "VisA_20220922").is_dir():
            self.root = self.root / "VisA_20220922"

    def categories(self):
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "Data" / "Images" / "Normal").is_dir()
        )

    def _image_dir(self, category, group):
        path = self.root / category / "Data" / "Images" / group
        if not path.is_dir():
            raise FileNotFoundError(f"missing VisA image directory: {path}")
        return path

    def samples(self, category, split):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        category_root = self.root / category
        if not category_root.is_dir():
            raise FileNotFoundError(f"missing VisA category: {category_root}")
        records = []
        normal_dir = self._image_dir(category, "Normal")
        for image_path in sorted(normal_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append({
                    "image_path": str(image_path),
                    "category": category,
                    "split": split,
                    "defect_type": "good",
                    "is_anomaly": False,
                    "mask_path": None,
                })
        if split == "test":
            anomaly_dir = self._image_dir(category, "Anomaly")
            mask_dir = category_root / "Data" / "Masks" / "Anomaly"
            for image_path in sorted(anomaly_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                mask_path = mask_dir / f"{image_path.stem}.png"
                records.append({
                    "image_path": str(image_path),
                    "category": category,
                    "split": split,
                    "defect_type": "anomaly",
                    "is_anomaly": True,
                    "mask_path": str(mask_path) if mask_path.exists() else None,
                })
        if not records:
            raise FileNotFoundError(f"no VisA images found for {category} ({split})")
        return records

    def split_counts(self, category, split):
        return dict(Counter(record["defect_type"] for record in self.samples(category, split)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    index = VisAIndex(args.root)
    for category in index.categories():
        print(category, index.split_counts(category, "train"), index.split_counts(category, "test"))


if __name__ == "__main__":
    main()
