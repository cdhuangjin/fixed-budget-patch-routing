"""CIFAR-100-C loading and fixed-latency checkpoint utilities."""

from pathlib import Path

import numpy as np
import torch


def load_cifar100c(root, corruption, severity):
    root = Path(root)
    image_path = root / f"{corruption}.npy"
    label_path = root / "labels.npy"
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"missing CIFAR-100-C files under {root}")
    images = np.load(image_path)
    labels = np.load(label_path)
    start = (int(severity) - 1) * 10000
    end = start + 10000
    if start < 0 or end > len(images) or len(labels) < 10000:
        raise ValueError("CIFAR-100-C corruption file must contain five 10000-image severities and 10000 labels")
    tensor_images = torch.from_numpy(np.asarray(images[start:end])).permute(0, 3, 1, 2).float().div(255.0)
    tensor_images = (tensor_images - torch.tensor((0.5071, 0.4867, 0.4408))[None, :, None, None]) / torch.tensor((0.2675, 0.2565, 0.2761))[None, :, None, None]
    return tensor_images, torch.from_numpy(np.asarray(labels[:10000], dtype=np.int64))


def select_p95_budget(points, target_p95_ms):
    if not points:
        raise ValueError("at least one validation latency point is required")
    feasible = [point for point in points if point["p95_ms"] <= target_p95_ms]
    if feasible:
        return max(feasible, key=lambda point: point["budget"])
    raise ValueError("no validation budget satisfies the fixed P95 latency target")
