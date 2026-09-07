"""Re-evaluate checkpoints with one shared, reference-defined hard subset."""

import argparse
import json
import re
from pathlib import Path

import torch

from real_cifar100 import evaluate_model, hard_example_indices, load_cifar100, make_loader
from smoke_train import build_model


def parse_checkpoint_path(path):
    path = Path(path)
    match = re.search(r"seed_(\d+)$", path.parent.parent.name)
    if path.name != "checkpoint.pt" or match is None:
        raise ValueError(f"unexpected checkpoint path: {path}")
    return {"seed": int(match.group(1)), "method": path.parent.name}


def collect_losses(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    model.to(device).eval()
    losses = []
    with torch.inference_mode():
        for images, labels in loader:
            logits, _ = model(images.to(device, non_blocking=True))
            losses.extend(criterion(logits, labels.to(device, non_blocking=True)).cpu().tolist())
    return losses


def load_model(checkpoint_path, method, device):
    model = build_model(method, image_size=32, num_classes=100)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-name", default="fixed_protocol_result.json")
    parser.add_argument("--reference-seed", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    runs_root = Path(args.runs_root)
    checkpoints = sorted(runs_root.glob("seed_*/**/checkpoint.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {runs_root}")
    reference_path = runs_root / f"seed_{args.reference_seed}" / "full" / "checkpoint.pt"
    if not reference_path.exists():
        raise FileNotFoundError(f"reference checkpoint missing: {reference_path}")

    dataset = load_cifar100(args.data_root, train=False, download=False)
    loader = make_loader(dataset, args.batch_size, shuffle=False, num_workers=2)
    reference_model = load_model(reference_path, "full", device)
    reference_losses = collect_losses(reference_model, loader, device)
    hard_indices = hard_example_indices(reference_losses)
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for checkpoint in checkpoints:
        meta = parse_checkpoint_path(checkpoint)
        model = load_model(checkpoint, meta["method"], device)
        metrics = evaluate_model(model, loader, device, hard_indices=hard_indices)
        result = {
            **meta,
            "reference_checkpoint": str(reference_path),
            "hard_subset_fraction": 0.10,
            "hard_subset_count": len(hard_indices),
            "evaluation": metrics,
        }
        Path(checkpoint.parent, args.output_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
