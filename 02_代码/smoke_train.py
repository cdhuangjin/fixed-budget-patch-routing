"""Small training harness for the 027 CIFAR-100 smoke pipeline."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from rata_vit import RATAConfig, RATAViT, risk_aware_loss


def make_synthetic_loader(samples, batch_size, image_size, num_classes, seed=5):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(samples, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, num_classes, (samples,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


def build_model(method, image_size=32, num_classes=100):
    patch_count = (image_size // 4) ** 2
    if method == "full":
        k_min = k_base = k_max = patch_count
        difficulty_gain = uncertainty_gain = 0.0
        adaptive = False
        route_policy = "full"
    elif method == "fixed_sparse":
        k_min = k_base = k_max = max(1, patch_count // 2)
        difficulty_gain = uncertainty_gain = 0.0
        adaptive = False
        route_policy = "fixed_sparse"
    elif method == "random_sparse":
        k_min = k_base = k_max = max(1, patch_count // 2)
        difficulty_gain = uncertainty_gain = 0.0
        adaptive = False
        route_policy = "random_sparse"
    elif method == "difficulty_only":
        k_min = max(1, patch_count // 4)
        k_base = max(k_min, patch_count // 2)
        k_max = patch_count
        difficulty_gain, uncertainty_gain = 1.0, 0.0
        adaptive = True
        route_policy = "difficulty_only"
    elif method == "uncertainty_only":
        k_min = max(1, patch_count // 4)
        k_base = max(k_min, patch_count // 2)
        k_max = patch_count
        difficulty_gain, uncertainty_gain = 0.0, 1.0
        adaptive = True
        route_policy = "uncertainty_only"
    elif method == "rata":
        k_min = max(1, patch_count // 4)
        k_base = max(k_min + 1, (patch_count * 3) // 8)
        k_max = max(k_base + 1, patch_count // 2)
        difficulty_gain = uncertainty_gain = 0.5
        adaptive = True
        route_policy = "rata"
    else:
        raise ValueError(f"unknown method: {method}")
    return RATAViT(
        RATAConfig(
            image_size=image_size,
            patch_size=4,
            num_classes=num_classes,
            embed_dim=32,
            depth=2,
            heads=4,
            k_min=k_min,
            k_base=k_base,
            k_max=k_max,
            difficulty_gain=difficulty_gain,
            uncertainty_gain=uncertainty_gain,
            adaptive=adaptive,
            route_policy=route_policy,
        )
    )


def train_steps(model, loader, steps, output_dir, device="cpu", lr=1e-3):
    device = torch.device(device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    iterator = iter(loader)
    for _ in range(steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        logits, route = model(images)
        loss = risk_aware_loss(logits, labels, route)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": steps}, output_path / "checkpoint.pt")
    result = {"steps_completed": steps, "loss": losses, "checkpoint": str(output_path / "checkpoint.pt")}
    (output_path / "history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("full", "fixed_sparse", "rata"), required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    model = build_model(args.method, args.image_size, args.num_classes)
    loader = make_synthetic_loader(args.batch_size * 2, args.batch_size, args.image_size, args.num_classes)
    result = train_steps(model, loader, args.steps, args.output_dir, device=device)
    result["method"] = args.method
    result["device"] = device
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
