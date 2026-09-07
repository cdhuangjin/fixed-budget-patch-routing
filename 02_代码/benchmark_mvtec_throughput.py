"""Batch throughput and tail-latency audit for the selective MVTec pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from evaluate_mvtec_patchcore import ResNetFeatures, collect_features, full_scores, split_normal_records
from mvtec_ad import MVTecADIndex


def make_batch(paths, size, norm, device):
    vals = []
    resize = transforms.Resize((size, size))
    to_tensor = transforms.ToTensor()
    for path in paths:
        vals.append(to_tensor(resize(Image.open(path).convert("RGB"))))
    return torch.stack(vals).to(device) if vals else torch.empty((0, 3, size, size), device=device)


def summarize(values):
    x = np.asarray(values, dtype=float)
    return {"n": int(x.size), "mean_ms_per_image": float(x.mean()),
            "p50_ms_per_image": float(np.percentile(x, 50)),
            "p95_ms_per_image": float(np.percentile(x, 95))}


def time_batch(fn, repeats, batch_size, device):
    for _ in range(2):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    vals = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        vals.append((time.perf_counter() - start) * 1000.0 / batch_size)
    return vals


def run_category(index, category, model, device, batch_size, repeats, seed):
    train = index.samples(category, "train")
    test = index.samples(category, "test")
    fit, _ = split_normal_records(train, seed)
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    fit_paths = [r["image_path"] for r in fit]
    test_paths = [r["image_path"] for r in test[: max(batch_size * 4, batch_size)]]
    fast_bank = torch.cat([model.forward_fast(make_batch([p], 128, norm, device)) for p in fit_paths], dim=0)
    full_bank = torch.cat([model.forward_full(make_batch([p], 224, norm, device))[0] for p in fit_paths], dim=0)
    local_bank = torch.cat([model.forward_fast_local(make_batch([p], 128, norm, device))[0] for p in fit_paths], dim=0)[::4]
    result = {}
    for start in range(0, len(test_paths), batch_size):
        paths = test_paths[start:start + batch_size]
        if len(paths) < batch_size:
            continue
        fast = make_batch(paths, 128, norm, device)
        full = make_batch(paths, 224, norm, device)

        def fast_fn():
            z = model.forward_fast(fast)
            torch.cdist(z, fast_bank).min(dim=1).values

        def full_fn():
            z = model.forward_full(full)
            full_scores(z, full_bank, device=device)

        def local_fn():
            z = model.forward_fast_local(fast)
            torch.cdist(z.reshape(-1, z.shape[-1]), local_bank).min(dim=1).values

        def mixed_fn():
            z = model.forward_fast_local(fast)
            local_dist = torch.cdist(z.reshape(-1, z.shape[-1]), local_bank).min(dim=1).values
            scores = local_dist.reshape(z.shape[0], z.shape[1]).mean(dim=1)
            k = max(1, int(np.ceil(batch_size * 0.25)))
            chosen = torch.topk(scores, k=k).indices
            model.forward_full(full.index_select(0, chosen))

        for name, fn in [("fast", fast_fn), ("full", full_fn), ("local_probe", local_fn), ("local_mixed", mixed_fn)]:
            result.setdefault(name, []).extend(time_batch(fn, repeats, batch_size, device))
    return {name: summarize(vals) for name, vals in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "capsule"])
    args = parser.parse_args()
    device = torch.device(args.device)
    model = ResNetFeatures(device)
    index = MVTecADIndex(args.data_root)
    out = {c: run_category(index, c, model, device, args.batch_size, args.repeats, args.seed) for c in args.categories}
    Path(args.output).write_text(json.dumps({"batch_size": args.batch_size, "repeats": args.repeats, "categories": out}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
