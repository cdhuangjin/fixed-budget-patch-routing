"""PaDiM-style diagonal Gaussian baseline on the same MVTec protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mvtec_ad import MVTecADIndex


def padim_scores(features, mean, variance, device):
    target = torch.device(device)
    features = features.to(target)
    mean = mean.to(target)
    variance = variance.to(target)
    distances = ((features - mean) ** 2 / variance).sum(dim=2).sqrt()
    k = max(1, int(round(distances.shape[1] * 0.1)))
    return torch.topk(distances, k=k, dim=1).values.mean(dim=1).detach().cpu().numpy()


def run_category(index, category, model, args, transform):
    from evaluate_mvtec_patchcore import auroc, collect_features, recall_at_fpr, split_normal_records

    train = index.samples(category, "train")
    test = index.samples(category, "test")
    fit_records, _ = split_normal_records(train, args.seed)
    fit, _, _ = collect_features(model, fit_records, 224, transform, args.batch_size, args.device, full=True)
    test_features, labels, _ = collect_features(model, test, 224, transform, args.batch_size, args.device, full=True)
    mean = fit.mean(dim=0)
    variance = fit.var(dim=0, unbiased=False).clamp_min(1e-4)
    scores = padim_scores(test_features, mean, variance, args.device)
    return {
        "category": category,
        "method": "padim_diagonal_gaussian",
        "image_auroc": auroc(labels, scores),
        "recall_at_fpr": recall_at_fpr(labels, scores),
        "n_fit": len(fit_records),
        "n_test": len(test),
        "feature_shape": list(fit.shape),
    }


def main():
    from evaluate_mvtec_patchcore import ResNetFeatures
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    model = ResNetFeatures(args.device)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    index = MVTecADIndex(args.data_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        result = run_category(index, category, model, args, transform)
        (output / f"{category}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
