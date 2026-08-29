"""VisA external validation for 027 - corrected version.
Uses combined scoring (fast for safe, augmented for risky) on FULL test set,
matching the MVTec evaluation protocol.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from evaluate_mvtec_patchcore import (
    ResNetFeatures, collect_features, collect_local_features, fast_scores, auroc,
    recall_at_fpr, split_normal_records
)
from selective_routes import (
    patch_memory_scores,
    patch_memory_dispersion_scores,
    quota_route,
    random_matched_route,
    score_matched_route,
    strict_quota_combined_scores,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class ImageRows(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        image = self.transform(Image.open(record["image_path"]).convert("RGB"))
        return image, index


def collect_full_per_image(model, records, transform, batch_size, device):
    loader = DataLoader(ImageRows(records, transform), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    features = []
    for images, _ in loader:
            features.extend(list(model.forward_full(images).detach().cpu()))
    return features


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
    p.add_argument("--route-local", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    local_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), normalize])

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
        local_bank = None
        if args.route_local:
            local_fit, _, _ = collect_local_features(model, train_normal, 128, local_transform, 32, device)
            local_bank = torch.cat(local_fit, dim=0)[::4]

        val_scores = fast_scores(val_features, bank)
        test_scores = fast_scores(test_features, bank)

        labels = [int(r["is_anomaly"]) for r in test_records]
        if args.route_local:
            local_test, _, _ = collect_local_features(model, test_records, 128, local_transform, 32, device)
            route_scores = patch_memory_scores(local_test, local_bank, 0.05, device).cpu().numpy()
            risk_route = quota_route(route_scores, args.fallback_budget)
            random_route = random_matched_route(len(route_scores), risk_route.actual_fallback_count, seed=args.seed)
            full_bank = torch.cat(collect_full_per_image(model, train_normal, transform, 32, device), dim=0)
            full_test = collect_full_per_image(model, test_records, transform, 32, device)
            full_patch_scores = patch_memory_scores(full_test, full_bank, 0.10, device).cpu().numpy()
            combined_risk_scores = np.where(risk_route.mask, full_patch_scores, test_scores)
            combined_random_scores = np.where(random_route.mask, full_patch_scores, test_scores)
            fast_route = score_matched_route(
                test_scores, risk_route.actual_fallback_count, route_name="fast_score"
            )
            uncertainty_scores = patch_memory_dispersion_scores(
                local_test, local_bank, device
            ).cpu().numpy()
            uncertainty_route = score_matched_route(
                uncertainty_scores, risk_route.actual_fallback_count,
                route_name="uncertainty_dispersion"
            )
            combined_fast_scores = np.where(fast_route.mask, full_patch_scores, test_scores)
            combined_uncertainty_scores = np.where(uncertainty_route.mask, full_patch_scores, test_scores)
            recalls = {
                "fast_only_recall": recall_at_fpr(labels, test_scores),
                "risk_combined_recall": recall_at_fpr(labels, combined_risk_scores),
                "random_combined_recall": recall_at_fpr(labels, combined_random_scores),
                "fast_score_combined_recall": recall_at_fpr(labels, combined_fast_scores),
                "uncertainty_combined_recall": recall_at_fpr(labels, combined_uncertainty_scores),
            }
            route_source = "test_free_exact_quota_local_patch_score_ranking"
        else:
            boost = 1.5
            combined_risk_scores, combined_random_scores, risk_route, random_route = strict_quota_combined_scores(
                test_scores, args.fallback_budget, args.seed, boost
            )
            route_source = "test_free_exact_quota_global_score_ranking"

        # Compute AUROC on FULL test set
        fast_auroc = float(auroc(labels, test_scores))
        risk_auroc = float(auroc(labels, combined_risk_scores))
        random_auroc = float(auroc(labels, combined_random_scores))

        result = {
            "category": cat,
            "fast_only_auroc": fast_auroc,
            "risk_combined_auroc": risk_auroc,
            "random_combined_auroc": random_auroc,
            "fast_score_combined_auroc": float(auroc(labels, combined_fast_scores)),
            "uncertainty_combined_auroc": float(auroc(labels, combined_uncertainty_scores)),
            **recalls,
            "risk_delta": risk_auroc - random_auroc,
            "budget": args.fallback_budget,
            "seed": args.seed,
            "fallback_rate": risk_route.actual_rate,
            "risk_count": risk_route.actual_fallback_count,
            "total": len(labels),
            "threshold": None,
            "route_local": bool(args.route_local),
            "route_source": route_source,
            "route": risk_route.as_dict(),
            "random_route": random_route.as_dict(),
            "fast_score_route": fast_route.as_dict(),
            "uncertainty_route": uncertainty_route.as_dict(),
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
