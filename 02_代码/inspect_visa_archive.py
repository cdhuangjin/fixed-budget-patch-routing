"""Inspect a VisA archive without extracting images or using a GPU."""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def inspect_archive(archive):
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    category_files = defaultdict(Counter)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) < 6 or parts[0] != "VisA_20220922":
                continue
            category = parts[1]
            if parts[2:5] == ("Data", "Images", "Normal"):
                category_files[category]["normal_images"] += 1
            elif parts[2:5] == ("Data", "Images", "Anomaly"):
                category_files[category]["anomaly_images"] += 1
            elif parts[2:5] == ("Data", "Masks", "Anomaly"):
                category_files[category]["anomaly_masks"] += 1
    categories = sorted(category_files)
    return {
        "archive": str(archive),
        "category_count": len(categories),
        "categories": categories,
        "per_category": {key: dict(category_files[key]) for key in categories},
        "normal_images": sum(category_files[key]["normal_images"] for key in categories),
        "anomaly_images": sum(category_files[key]["anomaly_images"] for key in categories),
        "anomaly_masks": sum(category_files[key]["anomaly_masks"] for key in categories),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect_archive(args.archive)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
