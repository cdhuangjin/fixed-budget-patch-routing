"""Evaluate all checkpoints on fixed CIFAR-100-C corruption subsets."""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from evaluate_cifar100c import load_cifar100c
from evaluate_fixed_protocol import collect_losses, load_model, parse_checkpoint_path
from real_cifar100 import evaluate_model, hard_example_indices, make_loader


def evaluate_corruption(runs_root, data_root, corruption, severity, output_root, batch_size=512, device="cuda"):
    runs_root = Path(runs_root)
    output_root = Path(output_root) / f"{corruption}_severity_{severity}"
    output_root.mkdir(parents=True, exist_ok=True)
    images, labels = load_cifar100c(Path(data_root) / "CIFAR-100-C", corruption, severity)
    loader = make_loader(TensorDataset(images, labels), batch_size, shuffle=False, num_workers=2)
    reference_path = runs_root / "seed_5" / "full" / "checkpoint.pt"
    reference_model = load_model(reference_path, "full", device)
    hard_indices = hard_example_indices(collect_losses(reference_model, loader, device))
    del reference_model

    for checkpoint in sorted(runs_root.glob("seed_*/**/checkpoint.pt")):
        meta = parse_checkpoint_path(checkpoint)
        model = load_model(checkpoint, meta["method"], device)
        metrics = evaluate_model(model, loader, device, hard_indices=hard_indices)
        result = {
            **meta,
            "corruption": corruption,
            "severity": int(severity),
            "reference_checkpoint": str(reference_path),
            "hard_subset_count": len(hard_indices),
            "evaluation": metrics,
        }
        target = output_root / f"seed_{meta['seed']}_{meta['method']}.json"
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--severity", type=int, choices=(1, 3, 5), required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate_corruption(**vars(args))


if __name__ == "__main__":
    main()
