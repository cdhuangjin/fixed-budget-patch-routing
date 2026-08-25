"""End-to-end batch-1 CUDA latency for Fast, Full and local-patch routes."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from evaluate_mvtec_patchcore import ResNetFeatures, split_normal_records
from mvtec_ad import MVTecADIndex
from analysis_protocol import summarize_latency


def image_tensor(path, size, norm):
    image = Image.open(path).convert("RGB")
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])(image).unsqueeze(0)


def full_score(features, bank):
    distances = torch.cdist(features[0], bank).min(dim=1).values
    return torch.topk(distances, k=max(1, int(round(len(distances) * 0.1)))).values.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", required=True)
    p.add_argument("--categories", nargs="+", default=None)
    p.add_argument("--bank-images", type=int, default=0,
                   help="normal reference images per category; 0 uses the formal fit split")
    p.add_argument("--repeats", type=int, default=100,
                   help="timed repetitions per cached image")
    args = p.parse_args()
    device = torch.device(args.device)
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    model = ResNetFeatures(device)
    index = MVTecADIndex(args.data_root)
    all_times = {"fast": [], "full": [], "risk": [], "local_probe": [], "local_risk": []}
    fallback_rates = []
    categories = args.categories or index.categories()
    for category in categories:
        train = index.samples(category, "train")
        test = index.samples(category, "test")
        fit, validation = split_normal_records(train, args.seed)
        # By default use the same fit split as the formal accuracy experiment.
        # A positive value is an explicit deployment-memory audit condition.
        if args.bank_images > 0:
            fit = fit[:args.bank_images]
        fast_bank = torch.cat([model.forward_fast(image_tensor(r["image_path"], 128, norm)) for r in fit], dim=0)
        full_bank = torch.cat([model.forward_full(image_tensor(r["image_path"], 224, norm))[0] for r in fit], dim=0)
        local_bank = torch.cat([
            model.forward_fast_local(image_tensor(r["image_path"], 128, norm))[0]
            for r in fit
        ], dim=0)[::4]
        validation_scores = torch.cat([
            torch.cdist(model.forward_fast(image_tensor(r["image_path"], 128, norm)), fast_bank).min(dim=1).values
            for r in validation
        ])
        threshold = torch.quantile(validation_scores, 0.75).item()
        local_validation_scores = torch.cat([
            torch.cdist(model.forward_fast_local(image_tensor(r["image_path"], 128, norm))[0], local_bank)
            .min(dim=1).values
            for r in validation
        ])
        local_threshold = torch.quantile(local_validation_scores, 0.75).item()
        risk_count = 0
        # Cache a fixed 5-image/category benchmark subset so the measured
        # P95 reflects inference rather than repeated disk decoding.
        cached = [
            (image_tensor(record["image_path"], 128, norm), image_tensor(record["image_path"], 224, norm))
            for record in test[:5]
        ]
        for fast_image, full_image in cached:
            # Warmup is performed once per route before collecting measurements.
            for route in ("fast", "full", "risk", "local_probe", "local_risk"):
                if route == "fast":
                    model.forward_fast(fast_image)
                elif route == "full":
                    model.forward_full(full_image)
                else:
                    model.forward_fast(fast_image)
            torch.cuda.synchronize() if device.type == "cuda" else None
            for _ in range(max(1, args.repeats)):
                start = time.perf_counter()
                z = model.forward_fast(fast_image)
                _ = torch.cdist(z, fast_bank).min()
                if device.type == "cuda": torch.cuda.synchronize()
                all_times["fast"].append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                zf = model.forward_full(full_image)
                _ = full_score(zf, full_bank)
                if device.type == "cuda": torch.cuda.synchronize()
                all_times["full"].append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                z = model.forward_fast(fast_image)
                score = torch.cdist(z, fast_bank).min()
                if score.item() >= threshold:
                    zf = model.forward_full(full_image)
                    _ = full_score(zf, full_bank)
                if device.type == "cuda": torch.cuda.synchronize()
                all_times["risk"].append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                zl = model.forward_fast_local(fast_image)
                local_score = torch.cdist(zl[0], local_bank).min(dim=1).values.topk(
                    max(1, int(round(zl.shape[1] * 0.05)))
                ).values.mean()
                if device.type == "cuda": torch.cuda.synchronize()
                all_times["local_probe"].append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                zl = model.forward_fast_local(fast_image)
                local_score = torch.cdist(zl[0], local_bank).min(dim=1).values.topk(
                    max(1, int(round(zl.shape[1] * 0.05)))
                ).values.mean()
                if local_score.item() >= local_threshold:
                    zf = model.forward_full(full_image)
                    _ = full_score(zf, full_bank)
                if device.type == "cuda": torch.cuda.synchronize()
                all_times["local_risk"].append((time.perf_counter() - start) * 1000.0)
            # Count fallback once per cached image, not once per timing repeat.
            risk_count += int(score.item() >= threshold)
        fallback_rates.append(risk_count / len(cached))
    result = {
        "n_images": len(all_times["fast"]),
        "fallback_rate_mean": float(np.mean(fallback_rates)),
        "fallback_rate_by_category": fallback_rates,
        "latency_protocol": {
            "batch_size": 1,
            "warmup_per_cached_image": 1,
            "repeats_per_cached_image": int(args.repeats),
            "cuda_synchronize": device.type == "cuda",
            "memory_bank_images_per_category": "formal_fit_split" if args.bank_images <= 0 else int(args.bank_images),
        },
        "latency_ms": {route: summarize_latency(values) for route, values in all_times.items()},
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
