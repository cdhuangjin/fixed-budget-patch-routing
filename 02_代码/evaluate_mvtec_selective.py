"""Validation-calibrated selective inference evaluation for MVTec AD.

The evaluator deliberately keeps calibration separate from test evaluation: only
normal training images are used to fit the normality scorer and choose the
fallback threshold.  The test split is then traversed once for reporting.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mvtec_ad import MVTecADIndex, split_good_images
from selective_inference import NormalityRiskScorer, choose_threshold, combine_risk, selective_predict
from smoke_train import build_model


class MVTecImageDataset(Dataset):
    def __init__(self, records, image_size=32):
        self.records = list(records)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).clone()
        image_tensor = values.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        return image_tensor, int(record["is_anomaly"]), index


def _predict(model, loader, device):
    model.eval()
    scores, features, uncertainties, labels = [], [], [], []
    elapsed = []
    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            start = time.perf_counter()
            logits, route = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000.0 / len(images))
            probabilities = logits.softmax(dim=-1)
            scores.extend((1.0 - probabilities[:, 0]).detach().cpu().tolist())
            features.append(route["pooled_feature"].detach().cpu())
            uncertainties.extend(route["uncertainty"].detach().cpu().tolist())
            labels.extend(batch_labels.tolist())
    return {
        "scores": torch.tensor(scores, dtype=torch.float32),
        "features": torch.cat(features, dim=0),
        "uncertainty": torch.tensor(uncertainties, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "latency_ms": elapsed,
    }


def binary_auroc(labels, scores):
    labels = [int(value) for value in labels]
    scores = [float(value) for value in scores]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = sum(rank + 1 for rank, index in enumerate(order) if labels[index] == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def recall_at_fpr(labels, scores, max_fpr=0.05):
    negatives = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    positives = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    if not negatives or not positives:
        return float("nan")
    threshold = sorted(negatives)[max(0, int(len(negatives) * (1.0 - max_fpr)) - 1)]
    return sum(score >= threshold for score in positives) / len(positives)


def _latency_summary(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"mean_ms": 0.0, "p95_latency_ms": 0.0}
    position = (len(values) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return {
        "mean_ms": sum(values) / len(values),
        "p95_latency_ms": values[lower] + (values[upper] - values[lower]) * (position - lower),
    }


def evaluate_records(train_records, validation_records, test_records, sparse_model, full_model,
                     device="cpu", batch_size=32, fallback_budget=0.25, seed=17):
    device = torch.device(device)
    train_loader = DataLoader(MVTecImageDataset(train_records), batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(MVTecImageDataset(validation_records), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MVTecImageDataset(test_records), batch_size=batch_size, shuffle=False)
    sparse_model.to(device)
    full_model.to(device)
    train_output = _predict(sparse_model, train_loader, device)
    validation_output = _predict(sparse_model, validation_loader, device)
    test_sparse = _predict(sparse_model, test_loader, device)
    test_full = _predict(full_model, test_loader, device)
    scorer = NormalityRiskScorer().fit(train_output["features"])
    validation_risk = combine_risk(
        scorer.score(validation_output["features"]), validation_output["uncertainty"]
    )
    threshold = choose_threshold(validation_risk.tolist(), fallback_budget)
    test_risk = combine_risk(scorer.score(test_sparse["features"]), test_sparse["uncertainty"])
    fast_predictions = {index: int(score >= 0.5) for index, score in enumerate(test_sparse["scores"])}
    full_predictions = {index: int(score >= 0.5) for index, score in enumerate(test_full["scores"])}
    risk_result = selective_predict(
        test_risk.tolist(), fast_predictions, lambda indices: {index: full_predictions[index] for index in indices},
        threshold, fast_latency_ms=sum(test_sparse["latency_ms"]) / max(1, len(test_sparse["latency_ms"])),
        full_latency_ms=sum(test_full["latency_ms"]) / max(1, len(test_full["latency_ms"])),
    )
    random_generator = random.Random(seed)
    fallback_count = risk_result["fallback_count"]
    random_indices = set(random_generator.sample(range(len(test_records)), fallback_count)) if fallback_count else set()
    random_predictions = dict(fast_predictions)
    for index in random_indices:
        random_predictions[index] = full_predictions[index]
    labels = test_sparse["labels"].tolist()
    selected_scores = test_sparse["scores"].tolist()
    random_scores = list(selected_scores)
    for index in random_indices:
        random_scores[index] = float(test_full["scores"][index])
    risk_indices = {index for index, flag in enumerate(risk_result["fallback"]) if flag}
    for index in risk_indices:
        selected_scores[index] = float(test_full["scores"][index])

    def metrics(predictions, scores, latency, fallback_rate=0.0):
        return {
            "image_auroc": binary_auroc(labels, scores),
            "recall_at_fpr": recall_at_fpr(labels, scores),
            "fallback_rate": fallback_rate,
            **_latency_summary(latency),
            "predictions": predictions,
        }

    sparse_latency = test_sparse["latency_ms"]
    full_latency = test_full["latency_ms"]
    selective_latency = [full_latency[min(i, len(full_latency) - 1)] if flag else sparse_latency[min(i, len(sparse_latency) - 1)]
                         for i, flag in enumerate(risk_result["fallback"])]
    random_latency = [full_latency[min(i, len(full_latency) - 1)] if i in random_indices else sparse_latency[min(i, len(sparse_latency) - 1)]
                      for i in range(len(test_records))]
    return {
        "threshold_source": "validation_normal_only",
        "threshold": float(threshold),
        "fallback_budget": float(fallback_budget),
        "full_only": metrics(full_predictions, test_full["scores"].tolist(), full_latency),
        "sparse_only": metrics(fast_predictions, test_sparse["scores"].tolist(), sparse_latency),
        "random_fallback": metrics(random_predictions, random_scores, random_latency, len(random_indices) / len(test_records)),
        "risk_fallback": metrics(risk_result["predictions"], selected_scores, selective_latency, risk_result["fallback_rate"]),
    }


def evaluate_category(data_root, category, device="cpu", batch_size=32, fallback_budget=0.25, seed=17,
                      sparse_checkpoint=None, full_checkpoint=None):
    index = MVTecADIndex(data_root)
    good_train = index.samples(category, "train")
    train_paths, validation_paths = split_good_images(
        [record["image_path"] for record in good_train], validation_fraction=0.1, seed=seed
    )
    train_paths, validation_paths = set(train_paths), set(validation_paths)
    train_records = [record for record in good_train if record["image_path"] in train_paths]
    validation_records = [record for record in good_train if record["image_path"] in validation_paths]
    sparse_model = build_model("fixed_sparse", image_size=32, num_classes=2)
    full_model = build_model("full", image_size=32, num_classes=2)
    if sparse_checkpoint:
        sparse_model.load_state_dict(torch.load(sparse_checkpoint, map_location="cpu"))
    if full_checkpoint:
        full_model.load_state_dict(torch.load(full_checkpoint, map_location="cpu"))
    result = evaluate_records(
        train_records, validation_records, index.samples(category, "test"), sparse_model, full_model,
        device=device, batch_size=batch_size, fallback_budget=fallback_budget, seed=seed,
    )
    result["category"] = category
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fallback-budget", type=float, default=0.25)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else None
        sparse_checkpoint = checkpoint_root / f"{category}_sparse.pt" if checkpoint_root else None
        full_checkpoint = checkpoint_root / f"{category}_full.pt" if checkpoint_root else None
        result = evaluate_category(
            args.data_root, category, args.device, args.batch_size, args.fallback_budget, seed=args.seed,
            sparse_checkpoint=sparse_checkpoint if sparse_checkpoint and sparse_checkpoint.exists() else None,
            full_checkpoint=full_checkpoint if full_checkpoint and full_checkpoint.exists() else None,
        )
        (output_root / f"{category}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(category, json.dumps({key: value for key, value in result.items() if key not in ("full_only", "sparse_only", "random_fallback", "risk_fallback")}))


if __name__ == "__main__":
    main()


