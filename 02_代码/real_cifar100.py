"""Real CIFAR-100 loading, short training, and unified evaluation."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from rata_vit import risk_aware_loss, summarize_latencies
from smoke_train import build_model


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hard_example_indices(reference_losses, fraction=0.10):
    count = max(1, math.ceil(len(reference_losses) * fraction))
    return sorted(range(len(reference_losses)), key=lambda index: reference_losses[index], reverse=True)[:count]


def summarize_hard_examples(losses, correct, fraction=0.10, hard_indices=None):
    indices = hard_example_indices(losses, fraction) if hard_indices is None else list(hard_indices)
    count = len(indices)
    return {"count": count, "accuracy": sum(bool(correct[index]) for index in indices) / count}


def compute_class_metrics(labels, predictions, num_classes):
    f1_scores = []
    counts = []
    for class_id in range(num_classes):
        true_positive = sum(label == class_id and prediction == class_id for label, prediction in zip(labels, predictions))
        false_positive = sum(label != class_id and prediction == class_id for label, prediction in zip(labels, predictions))
        false_negative = sum(label == class_id and prediction != class_id for label, prediction in zip(labels, predictions))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        counts.append(sum(label == class_id for label in labels))
    return {"macro_f1": sum(f1_scores) / num_classes, "class_counts": counts}


def compute_ece(confidences, correct, bins=10):
    total = len(confidences)
    if total == 0:
        return 0.0
    error = 0.0
    for bin_id in range(bins):
        lower = bin_id / bins
        upper = (bin_id + 1) / bins
        members = [index for index, confidence in enumerate(confidences) if lower <= confidence <= upper and (bin_id == bins - 1 or confidence < upper)]
        if members:
            accuracy = sum(bool(correct[index]) for index in members) / len(members)
            confidence = sum(confidences[index] for index in members) / len(members)
            error += len(members) / total * abs(accuracy - confidence)
    return error


def make_loader(dataset, batch_size, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_cifar100(data_root, train, download=False):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return datasets.CIFAR100(root=data_root, train=train, transform=transform, download=download)


def evaluate_model(model, loader, device, hard_indices=None):
    device = torch.device(device)
    model.to(device).eval()
    losses, correct, labels_all, predictions_all, confidences, latencies = [], [], [], [], [], []
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits, _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000.0)
            losses.extend(criterion(logits, labels).cpu().tolist())
            predictions = logits.argmax(dim=1)
            correct.extend((predictions == labels).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            confidences.extend(logits.softmax(dim=1).max(dim=1).values.cpu().tolist())
    hard = summarize_hard_examples(losses, correct, hard_indices=hard_indices)
    class_metrics = compute_class_metrics(labels_all, predictions_all, logits.shape[-1])
    return {
        "samples": len(losses),
        "accuracy": sum(correct) / len(correct),
        "worst_10pct_accuracy": hard["accuracy"],
        "hard_example_count": hard["count"],
        "mean_loss": sum(losses) / len(losses),
        **class_metrics,
        "ece": compute_ece(confidences, correct),
        "latency": summarize_latencies(latencies),
        "p95_latency_ms": summarize_latencies(latencies)["p95_ms"],
    }


def train_model(model, loader, steps, device, output_dir):
    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    iterator = iter(loader)
    for _ in range(steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits, route = model(images)
        loss = risk_aware_loss(logits, labels, route)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": steps}, output_path / "checkpoint.pt")
    return {"steps_completed": steps, "first_loss": losses[0], "last_loss": losses[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("full", "fixed_sparse", "random_sparse", "difficulty_only", "uncertainty_only", "rata"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    train_set = load_cifar100(args.data_root, train=True, download=args.download)
    test_set = load_cifar100(args.data_root, train=False, download=args.download)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=2)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False, num_workers=2)
    model = build_model(args.method, image_size=32, num_classes=100)
    train_result = train_model(model, train_loader, args.steps, device, args.output_dir)
    metrics = evaluate_model(model, test_loader, device)
    result = {"method": args.method, "seed": args.seed, "device": device, "train": train_result, "evaluation": metrics}
    Path(args.output_dir, "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
