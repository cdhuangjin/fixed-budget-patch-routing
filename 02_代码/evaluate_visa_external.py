"""VisA external validation for 027 - corrected version.
Uses combined scoring (fast for safe, augmented for risky) on FULL test set,
matching the MVTec evaluation protocol.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from evaluate_mvtec_patchcore import (
    ResNetFeatures, collect_features, fast_scores, auroc, split_normal_records
)
from selective_routes import strict_quota_combined_scores

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class VisAIndex:
    def __init__(self, root):
        self.root = Path(root)
        if (self.root / "VisA_20220922").is_dir():
            self.root = self.root / "VisA_20220922"

    def categories(self):
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / "Data" / "Images" / "Normal").is_dir()
        )

    def samples(self, category, split):
        cat_root = self.root / category / "Data" / "Images"
        records = []
        for img in sorted((cat_root / "Normal").iterdir()):
            if img.suffix.lower() in IMAGE_SUFFIXES:
                records.append({"image_path": str(img), "label": "good", "is_anomaly": False})
        if split == "test":
            for img in sorted((cat_root / "Anomaly").iterdir()):
                if img.suffix.lower() in IMAGE_SUFFIXES:
                    records.append({"image_path": str(img), "label": "anomaly", "is_anomaly": True})
        return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--fallback-budget", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    index = VisAIndex(args.data_root)
    model = ResNetFeatures(device)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for cat in args.categories:
        train_records = index.samples(cat, "train")
        test_records = index.samples(cat, "test")
        train_normal, val_normal = split_normal_records(train_records, seed=args.seed)

        train_features, _, fast_ms = collect_features(model, train_normal, 224, transform, 32, device)
        val_features, _, _ = collect_features(model, val_normal, 224, transform, 32, device)
        test_features, _, _ = collect_features(model, test_records, 224, transform, 32, device)

        bank = train_features

        val_scores = fast_scores(val_features, bank)
        test_scores = fast_scores(test_features, bank)

        labels = [int(r["is_anomaly"]) for r in test_records]
        boost = 1.5
        combined_risk_scores, combined_random_scores, risk_route, random_route = strict_quota_combined_scores(
            test_scores, args.fallback_budget, args.seed, boost
        )

        # Compute AUROC on FULL test set
        fast_auroc = float(auroc(labels, test_scores))
        risk_auroc = float(auroc(labels, combined_risk_scores))
        random_auroc = float(auroc(labels, combined_random_scores))

        result = {
            "category": cat,
            "fast_only_auroc": fast_auroc,
            "risk_combined_auroc": risk_auroc,
            "random_combined_auroc": random_auroc,
            "risk_delta": risk_auroc - random_auroc,
            "budget": args.fallback_budget,
            "seed": args.seed,
            "fallback_rate": risk_route.actual_rate,
            "risk_count": risk_route.actual_fallback_count,
            "total": len(labels),
            "threshold": None,
            "route": risk_route.as_dict(),
            "random_route": random_route.as_dict(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    fast_avg = np.mean([r["fast_only_auroc"] for r in results])
    risk_avg = np.mean([r["risk_combined_auroc"] for r in results])
    random_avg = np.mean([r["random_combined_auroc"] for r in results])
    delta_avg = np.mean([r["risk_delta"] for r in results])
    print(f"\nFast avg: {fast_avg:.4f}")
    print(f"Risk combined avg: {risk_avg:.4f}")
    print(f"Random combined avg: {random_avg:.4f}")
    print(f"Risk-Random delta avg: {delta_avg:+.4f}")
    print(f"Risk > Random in {sum(1 for r in results if r['risk_delta'] > 0)}/{len(results)} categories")


if __name__ == "__main__":
    main()
