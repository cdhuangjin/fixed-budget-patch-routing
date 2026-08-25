"""Reproducible batch-1 CUDA P50/P95 feature-path latency benchmark."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from evaluate_mvtec_patchcore import ResNetFeatures
from mvtec_ad import MVTecADIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = ResNetFeatures(device)
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    index = MVTecADIndex(args.data_root)
    samples = []
    for category in index.categories():
        record = index.samples(category, "test")[0]
        image = Image.open(record["image_path"]).convert("RGB")
        fast = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), norm])(image).unsqueeze(0)
        full = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])(image).unsqueeze(0)
        samples.append((category, fast, full))
    result = {}
    for mode in ("fast", "full"):
        times = []
        for _, fast, full in samples:
            tensor = fast if mode == "fast" else full
            for _ in range(20):
                (model.forward_fast(tensor) if mode == "fast" else model.forward_full(tensor))
            if device.type == "cuda":
                torch.cuda.synchronize()
            for _ in range(args.repeats):
                start = time.perf_counter()
                model.forward_fast(tensor) if mode == "fast" else model.forward_full(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
        result[mode] = {
            "n": len(times), "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times)),
        }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
