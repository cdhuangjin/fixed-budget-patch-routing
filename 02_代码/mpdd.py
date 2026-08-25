"""MPDD directory indexing for normal-only industrial anomaly evaluation."""

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


class MPDDIndex:
    """Index either the extracted MPDD root or its nested MPDD/MPDD layout."""

    def __init__(self, root):
        self.root = Path(root)
        if (self.root / "MPDD").is_dir() and not (self.root / "bracket_black").is_dir():
            self.root = self.root / "MPDD"
        if not self.root.exists():
            raise FileNotFoundError(f"MPDD root does not exist: {self.root}")

    def categories(self):
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        )

    def samples(self, category, split):
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        category_root = self.root / category
        split_root = category_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing MPDD split: {split_root}")
        records = []
        for defect_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            is_anomaly = defect_root.name != "good"
            for image_path in sorted(
                path for path in defect_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
            ):
                mask_path = None
                if is_anomaly:
                    candidate = category_root / "ground_truth" / defect_root.name / f"{image_path.stem}_mask.png"
                    mask_path = str(candidate) if candidate.exists() else None
                records.append({
                    "image_path": str(image_path),
                    "category": category,
                    "split": split,
                    "defect_type": defect_root.name,
                    "is_anomaly": is_anomaly,
                    "mask_path": mask_path,
                })
        if not records:
            raise FileNotFoundError(f"no images found under {split_root}")
        return records

    def split_counts(self, category, split):
        return dict(Counter(record["defect_type"] for record in self.samples(category, split)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    index = MPDDIndex(args.root)
    for category in index.categories():
        print(category, index.split_counts(category, "train"), index.split_counts(category, "test"))


if __name__ == "__main__":
    main()
