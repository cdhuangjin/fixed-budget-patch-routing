#!/usr/bin/env python3
"""Measure end-to-end latency, including routing and token selection."""

import argparse
import json
import time
from pathlib import Path

import torch

from rata_vit import RATAConfig, RATAViT, summarize_latencies
from smoke_train import build_model


def measure(model, images, warmup, repeats, device):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000.0)
    return summarize_latencies(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--method", choices=("full", "fixed_sparse", "random_sparse", "difficulty_only", "uncertainty_only", "rata"), default="rata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    if args.image_size == 32:
        model = build_model(args.method, image_size=32, num_classes=100).to(device)
    else:
        patch_count = (args.image_size // 16) ** 2
        if args.method == "full":
            k_min = k_base = k_max = patch_count
            adaptive = False
        elif args.method == "fixed_sparse":
            k_min = k_base = k_max = max(1, patch_count // 2)
            adaptive = False
        else:
            k_min = max(1, patch_count // 4)
            k_base = max(k_min + 1, (patch_count * 3) // 8)
            k_max = max(k_base + 1, patch_count // 2)
            adaptive = True
        config = RATAConfig(
            image_size=args.image_size,
            k_min=k_min,
            k_base=k_base,
            k_max=k_max,
            adaptive=adaptive,
        )
        model = RATAViT(config).to(device)
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "method": args.method,
        "latency": measure(model, images, args.warmup, args.repeats, device),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
