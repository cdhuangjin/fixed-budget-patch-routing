"""Pretrained-feature selective anomaly detection on MVTec AD.

This is the application-paper protocol for 027.  The fast route uses a
low-resolution ResNet-18 layer-2 global embedding; the full route uses a
higher-resolution layer-3 patch memory bank (PatchCore-style nearest-neighbor
scoring).  The router is calibrated on held-out normal images only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms

from mvtec_ad import MVTecADIndex
from selective_routes import (
    oracle_matched_route,
    random_matched_route,
    risk_route,
    score_matched_route,
    quota_route,
    patch_memory_dispersion_scores,
    online_prefix_quota_route,
    threshold_for_budget,
    conformal_threshold,
)


class ImageRecords(Dataset):
    def __init__(self, records: list[dict], size: int, transform, flip=False):
        self.records = list(records)
        self.size = size
        self.transform = transform
        self.flip = bool(flip)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        if self.flip:
            image = ImageOps.mirror(image)
        return self.transform(image), int(record["is_anomaly"]), index


def split_normal_records(records, seed=17):
    normal = [r for r in records if not r["is_anomaly"]]
    rng = random.Random(seed)
    rng.shuffle(normal)
    cut = max(1, int(round(len(normal) * 0.8)))
    return normal[:cut], normal[cut:]


class ResNetFeatures(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)
        self.stem = torch.nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.device = torch.device(device)
        self.to(self.device).eval()

    @torch.inference_mode()
    def forward_fast(self, images):
        x = images.to(self.device, non_blocking=True)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return F.normalize(x.mean(dim=(2, 3)), dim=1)

    @torch.inference_mode()
    def forward_fast_local(self, images):
        x = images.to(self.device, non_blocking=True)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return F.normalize(x.flatten(2).transpose(1, 2), dim=2)

    @torch.inference_mode()
    def forward_full(self, images):
        x = images.to(self.device, non_blocking=True)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # PatchCore-style local descriptors, normalized per patch.
        patches = x.flatten(2).transpose(1, 2)
        return F.normalize(patches, dim=2)


def collect_features(model, records, size, transform, batch_size, device, full=False, flip=False):
    loader = DataLoader(ImageRecords(records, size, transform, flip=flip), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    all_features, labels = [], []
    start = time.perf_counter()
    for images, batch_labels, _ in loader:
        features = model.forward_full(images) if full else model.forward_fast(images)
        all_features.append(features.detach().cpu())
        labels.extend(batch_labels.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return torch.cat(all_features), np.asarray(labels), elapsed / max(1, len(records)) * 1000.0


def collect_local_features(model, records, size, transform, batch_size, device, flip=False):
    loader = DataLoader(ImageRecords(records, size, transform, flip=flip), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    all_features, labels = [], []
    start = time.perf_counter()
    for images, batch_labels, _ in loader:
        features = model.forward_fast_local(images).detach().cpu()
        all_features.extend(list(features))
        labels.extend(batch_labels.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return all_features, np.asarray(labels), elapsed / max(1, len(records)) * 1000.0


def fast_scores(features, bank, device=None):
    target = torch.device(device) if device is not None else features.device
    distances = torch.cdist(features.to(target), bank.to(target))
    return distances.min(dim=1).values.detach().cpu().numpy()


def fast_topk_scores(features, bank, topk=1, device=None):
    """Return a robust matching cost from the k nearest normal prototypes."""
    target = torch.device(device) if device is not None else features.device
    distances = torch.cdist(features.to(target), bank.to(target))
    k = min(max(1, int(topk)), distances.shape[1])
    return torch.topk(distances, k=k, largest=False, dim=1).values.mean(dim=1).detach().cpu().numpy()


def fast_top2_scores(features, bank, device=None):
    """Return the two nearest-neighbour distances for ambiguity-aware routing."""
    target = torch.device(device) if device is not None else features.device
    distances = torch.cdist(features.to(target), bank.to(target))
    top2 = torch.topk(distances, k=min(2, distances.shape[1]), largest=False, dim=1).values
    if top2.shape[1] == 1:
        top2 = torch.cat([top2, top2], dim=1)
    return top2[:, 0].detach().cpu().numpy(), top2[:, 1].detach().cpu().numpy()


def full_scores(features, bank, top_fraction=0.1, device=None):
    target = torch.device(device) if device is not None else features.device
    bank = bank.to(target)
    scores = []
    for sample in features:
        distances = torch.cdist(sample.to(target), bank)
        nearest = distances.min(dim=1).values
        k = max(1, int(round(len(nearest) * top_fraction)))
        scores.append(torch.topk(nearest, k=k).values.mean())
    return torch.stack(scores).detach().cpu().numpy()


def image_coreset_bank(full_features, ratio: float, seed: int = 17):
    """Select representative normal images, retaining all patches per image."""
    features = full_features.detach().cpu()
    n_images = int(features.shape[0])
    target = max(1, min(n_images, int(np.ceil(n_images * float(ratio)))))
    if target >= n_images:
        return features.reshape(-1, features.shape[-1]), np.arange(n_images)
    pooled = F.normalize(features.mean(dim=1), dim=1)
    rng = np.random.default_rng(int(seed))
    selected = [int(rng.integers(0, n_images))]
    min_dist = torch.full((n_images,), float("inf"))
    for _ in range(1, target):
        last = pooled[selected[-1]].unsqueeze(0)
        dist = torch.cdist(pooled, last).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
        min_dist[selected] = -1.0
        selected.append(int(torch.argmax(min_dist).item()))
    indices = np.asarray(selected, dtype=int)
    return features[indices].reshape(-1, features.shape[-1]), indices


def auroc(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(y_true, scores, fpr=0.05):
    y = np.asarray(y_true, dtype=int)
    normal = np.sort(np.asarray(scores)[y == 0])
    threshold = normal[min(len(normal) - 1, int(np.floor((1 - fpr) * len(normal))))]
    return float(np.asarray(scores)[y == 1].__ge__(threshold).mean())


def empirical_rank_scores(validation_scores, query_scores):
    """Map scores to validation-only empirical percentiles for rank fusion."""
    reference = np.sort(np.asarray(validation_scores, dtype=float).reshape(-1))
    query = np.asarray(query_scores, dtype=float).reshape(-1)
    return np.searchsorted(reference, query, side="right") / max(1, len(reference))


def evaluate_mode(y_true, scores, fallback_rate, fast_ms, full_ms):
    return {
        "image_auroc": auroc(y_true, scores),
        "recall_at_fpr": recall_at_fpr(y_true, scores),
        "fallback_rate": float(fallback_rate),
        "mean_ms_estimated": float(fast_ms + fallback_rate * full_ms),
        "p95_ms_estimated": float(fast_ms + fallback_rate * full_ms),
    }


def run_category(index, category, args, model, transform):
    train = index.samples(category, "train")
    test = index.samples(category, "test")
    fit_records, validation_records = split_normal_records(train, args.seed)
    fast_fit, _, fast_ms = collect_features(model, fit_records, args.fast_size, transform, args.batch_size, args.device)
    fast_val, _, _ = collect_features(model, validation_records, args.fast_size, transform, args.batch_size, args.device)
    fast_test, y_test, _ = collect_features(model, test, args.fast_size, transform, args.batch_size, args.device)
    local_val_scores = local_test_scores = None
    local_bank = None
    local_ms = 0.0
    if bool(getattr(args, "route_local", False)):
        local_fit, _, local_ms = collect_local_features(model, fit_records, args.fast_size, transform, args.batch_size, args.device)
        local_val, _, _ = collect_local_features(model, validation_records, args.fast_size, transform, args.batch_size, args.device)
        local_test, _, _ = collect_local_features(model, test, args.fast_size, transform, args.batch_size, args.device)
        local_bank = torch.cat(local_fit, dim=0)[::4]
        local_top_fraction = float(getattr(args, "local_top_fraction", 0.05))
        local_val_scores = full_scores(local_val, local_bank, top_fraction=local_top_fraction, device=args.device)
        local_test_scores = full_scores(local_test, local_bank, top_fraction=local_top_fraction, device=args.device)
        if bool(getattr(args, "local_flip", False)):
            local_val_flip, _, _ = collect_local_features(model, validation_records, args.fast_size, transform, args.batch_size, args.device, flip=True)
            local_test_flip, _, _ = collect_local_features(model, test, args.fast_size, transform, args.batch_size, args.device, flip=True)
            local_val_flip_scores = full_scores(local_val_flip, local_bank, top_fraction=local_top_fraction, device=args.device)
            local_test_flip_scores = full_scores(local_test_flip, local_bank, top_fraction=local_top_fraction, device=args.device)
            local_val_scores = np.maximum(local_val_scores, local_val_flip_scores)
            local_test_scores = np.maximum(local_test_scores, local_test_flip_scores)
    route_ms = int(getattr(args, "route_ms", 0))
    medium_val_scores = medium_test_scores = None
    medium_ms = 0.0
    if route_ms > args.fast_size:
        medium_fit, _, medium_ms = collect_features(
            model, fit_records, route_ms, transform, args.batch_size, args.device
        )
        medium_val, _, _ = collect_features(
            model, validation_records, route_ms, transform, args.batch_size, args.device
        )
        medium_test, _, _ = collect_features(
            model, test, route_ms, transform, args.batch_size, args.device
        )
        medium_bank = medium_fit
        medium_val_raw = fast_scores(medium_val, medium_bank, device=args.device)
        medium_test_raw = fast_scores(medium_test, medium_bank, device=args.device)
        fast_val_raw = fast_scores(fast_val, fast_fit, device=args.device)
        fast_test_raw = fast_scores(fast_test, fast_fit, device=args.device)
        fast_val_scores = 0.5 * empirical_rank_scores(fast_val_raw, fast_val_raw) + 0.5 * empirical_rank_scores(medium_val_raw, medium_val_raw)
        fast_test_scores = 0.5 * empirical_rank_scores(fast_val_raw, fast_test_raw) + 0.5 * empirical_rank_scores(medium_val_raw, medium_test_raw)
    full_transform = getattr(args, "_full_transform", transform)
    full_fit, _, full_ms = collect_features(model, fit_records, args.full_size, full_transform, args.batch_size, args.device, full=True)
    full_test, _, _ = collect_features(model, test, args.full_size, full_transform, args.batch_size, args.device, full=True)
    fast_bank = fast_fit
    full_bank = full_fit.reshape(-1, full_fit.shape[-1])
    full_bank_indices = None
    if getattr(args, "full_bank_mode", "all") == "image_coreset":
        full_bank, full_bank_indices = image_coreset_bank(
            full_fit, getattr(args, "full_bank_ratio", 1.0), seed=args.seed
        )
    fast_val_scores = fast_scores(fast_val, fast_bank, device=args.device)
    fast_test_scores = fast_scores(fast_test, fast_bank, device=args.device)
    if route_ms > args.fast_size:
        # Recompute the route score after the standard Fast bank is fixed;
        # percentile fusion changes the ordering while remaining label-free.
        fast_val_raw = fast_val_scores
        fast_test_raw = fast_test_scores
        fast_val_scores = 0.5 * empirical_rank_scores(fast_val_raw, fast_val_raw) + 0.5 * empirical_rank_scores(medium_val_raw, medium_val_raw)
        fast_test_scores = 0.5 * empirical_rank_scores(fast_val_raw, fast_test_raw) + 0.5 * empirical_rank_scores(medium_val_raw, medium_test_raw)
    route_flip = bool(getattr(args, "route_flip", False))
    if route_flip:
        fast_val_flip, _, _ = collect_features(
            model, validation_records, args.fast_size, transform, args.batch_size, args.device, flip=True
        )
        fast_test_flip, _, _ = collect_features(
            model, test, args.fast_size, transform, args.batch_size, args.device, flip=True
        )
        # Max pooling is a label-free risk proxy: samples unstable under a
        # horizontal reflection receive the more conservative Fast score.
        fast_val_scores = np.maximum(fast_val_scores, fast_scores(fast_val_flip, fast_bank, device=args.device))
        fast_test_scores = np.maximum(fast_test_scores, fast_scores(fast_test_flip, fast_bank, device=args.device))
    route_topk = max(1, int(getattr(args, "route_topk", 1)))
    route_val_scores = fast_val_scores
    route_test_scores = fast_test_scores
    if local_val_scores is not None:
        route_val_scores = local_val_scores
        route_test_scores = local_test_scores
        local_global_alpha = float(getattr(args, "local_global_alpha", 0.0))
        if local_global_alpha:
            alpha = min(1.0, max(0.0, local_global_alpha))
            local_val_rank = empirical_rank_scores(local_val_scores, local_val_scores)
            local_test_rank = empirical_rank_scores(local_val_scores, local_test_scores)
            fast_val_rank = empirical_rank_scores(fast_val_scores, fast_val_scores)
            fast_test_rank = empirical_rank_scores(fast_val_scores, fast_test_scores)
            route_val_scores = (1.0 - alpha) * local_val_rank + alpha * fast_val_rank
            route_test_scores = (1.0 - alpha) * local_test_rank + alpha * fast_test_rank
    if route_topk > 1:
        route_val_scores = fast_topk_scores(fast_val, fast_bank, route_topk, device=args.device)
        route_test_scores = fast_topk_scores(fast_test, fast_bank, route_topk, device=args.device)
    gap_alpha = float(getattr(args, "gap_alpha", 0.0))
    gap_val_scores = gap_test_scores = None
    if gap_alpha:
        val_d1, val_d2 = fast_top2_scores(fast_val, fast_bank, device=args.device)
        test_d1, test_d2 = fast_top2_scores(fast_test, fast_bank, device=args.device)
        # The gap is an ambiguity signal: a large second-neighbour margin means
        # the query is not well explained by a local normal prototype.
        gap_val_scores = val_d1 + gap_alpha * (val_d2 - val_d1)
        gap_test_scores = test_d1 + gap_alpha * (test_d2 - test_d1)
    full_test_scores = full_scores(full_test, full_bank, device=args.device)
    # Validation-only threshold; no anomalous validation labels are used.
    threshold = threshold_for_budget(route_val_scores, args.fallback_budget)
    risk_decision = risk_route(route_test_scores, threshold)
    random_decision = random_matched_route(
        len(test), risk_decision.actual_fallback_count, seed=args.seed
    )
    score_decision = score_matched_route(
        route_test_scores, risk_decision.actual_fallback_count, route_name="score_matched"
    )
    quota_decision = quota_route(route_test_scores, args.fallback_budget)
    quota_random_decision = random_matched_route(
        len(test), quota_decision.actual_fallback_count, seed=args.seed
    )
    online_batch_size = max(1, int(getattr(args, "online_route_batch_size", args.batch_size)))
    online_decision = online_prefix_quota_route(
        route_test_scores, args.fallback_budget, batch_size=online_batch_size
    )
    online_random_decision = random_matched_route(
        len(test), online_decision.actual_fallback_count, seed=args.seed
    )
    conformal_threshold_value = conformal_threshold(route_val_scores, args.fallback_budget)
    conformal_decision = risk_route(route_test_scores, conformal_threshold_value)
    conformal_random_decision = random_matched_route(
        len(test), conformal_decision.actual_fallback_count, seed=args.seed
    )
    oracle_decision = oracle_matched_route(
        y_test, risk_decision.actual_fallback_count
    )
    risk_mask = risk_decision.mask
    random_mask = random_decision.mask
    fallback_rate = risk_decision.actual_rate
    combined_risk = np.where(risk_mask, full_test_scores, fast_test_scores)
    combined_random = np.where(random_mask, full_test_scores, fast_test_scores)
    combined_score = np.where(score_decision.mask, full_test_scores, fast_test_scores)
    combined_quota = np.where(quota_decision.mask, full_test_scores, fast_test_scores)
    fast_score_decision = score_matched_route(
        fast_test_scores, quota_decision.actual_fallback_count, route_name="fast_score"
    )
    if local_test_scores is None or local_bank is None:
        raise ValueError("the matched uncertainty baseline requires local route features")
    uncertainty_test_scores = patch_memory_dispersion_scores(
        local_test, local_bank, device=args.device
    ).cpu().numpy()
    uncertainty_decision = score_matched_route(
        uncertainty_test_scores, quota_decision.actual_fallback_count,
        route_name="uncertainty_dispersion"
    )
    combined_fast_score = np.where(
        fast_score_decision.mask, full_test_scores, fast_test_scores
    )
    combined_uncertainty = np.where(
        uncertainty_decision.mask, full_test_scores, fast_test_scores
    )
    combined_quota_random = np.where(
        quota_random_decision.mask, full_test_scores, fast_test_scores
    )
    combined_online = np.where(
        online_decision.mask, full_test_scores, fast_test_scores
    )
    combined_online_random = np.where(
        online_random_decision.mask, full_test_scores, fast_test_scores
    )
    combined_conformal = np.where(
        conformal_decision.mask, full_test_scores, fast_test_scores
    )
    combined_conformal_random = np.where(
        conformal_random_decision.mask, full_test_scores, fast_test_scores
    )
    combined_oracle = np.where(oracle_decision.mask, full_test_scores, fast_test_scores)
    normal_mask = y_test == 0
    anomaly_mask = y_test == 1
    result = {
        "category": category,
        "threshold_source": "heldout_normal_only",
        "threshold": threshold,
        "fallback_budget": args.fallback_budget,
        "route_topk": route_topk,
        "route_flip": route_flip,
        "route_ms": route_ms,
        "route_medium_ms": float(medium_ms),
        "route_local": bool(getattr(args, "route_local", False)),
        "local_top_fraction": float(getattr(args, "local_top_fraction", 0.05)),
        "local_global_alpha": float(getattr(args, "local_global_alpha", 0.0)),
        "local_flip": bool(getattr(args, "local_flip", False)),
        "full_bank_mode": getattr(args, "full_bank_mode", "all"),
        "full_bank_ratio": float(getattr(args, "full_bank_ratio", 1.0)),
        "full_bank_images": int(len(full_bank_indices)) if full_bank_indices is not None else int(full_fit.shape[0]),
        "route_local_ms": float(local_ms),
        "fast_only": evaluate_mode(y_test, fast_test_scores, 0.0, fast_ms, full_ms),
        "full_only": evaluate_mode(y_test, full_test_scores, 1.0, fast_ms, full_ms),
        "random_fallback": evaluate_mode(y_test, combined_random, float(random_mask.mean()), fast_ms, full_ms),
        "risk_fallback": evaluate_mode(y_test, combined_risk, fallback_rate, fast_ms, full_ms),
        "score_matched_upper": evaluate_mode(
            y_test, combined_score, score_decision.actual_rate, fast_ms, full_ms
        ),
        "fast_score": evaluate_mode(
            y_test, combined_fast_score, fast_score_decision.actual_rate, fast_ms, full_ms
        ),
        "uncertainty_dispersion": evaluate_mode(
            y_test, combined_uncertainty, uncertainty_decision.actual_rate, fast_ms, full_ms
        ),
        "strict_quota": evaluate_mode(
            y_test, combined_quota, quota_decision.actual_rate, fast_ms, full_ms
        ),
        "strict_quota_random": evaluate_mode(
            y_test, combined_quota_random, quota_random_decision.actual_rate, fast_ms, full_ms
        ),
        "online_prefix_quota": evaluate_mode(
            y_test, combined_online, online_decision.actual_rate, fast_ms, full_ms
        ),
        "online_prefix_quota_random": evaluate_mode(
            y_test, combined_online_random, online_random_decision.actual_rate, fast_ms, full_ms
        ),
        "conformal_local": evaluate_mode(
            y_test, combined_conformal, conformal_decision.actual_rate, fast_ms, full_ms
        ),
        "conformal_local_random": evaluate_mode(
            y_test, combined_conformal_random, conformal_random_decision.actual_rate, fast_ms, full_ms
        ),
        "oracle_label_upper": evaluate_mode(
            y_test, combined_oracle, oracle_decision.actual_rate, fast_ms, full_ms
        ),
        "operational_fallback": {
            "random_normal": float(random_mask[normal_mask].mean()),
            "risk_normal": float(risk_mask[normal_mask].mean()),
            "random_anomaly": float(random_mask[anomaly_mask].mean()),
            "risk_anomaly": float(risk_mask[anomaly_mask].mean()),
            "fast_feature_ms_per_image": float(fast_ms),
            "full_feature_ms_per_image": float(full_ms),
        },
        "routing": {
            "risk": risk_decision.as_dict(),
            "random": random_decision.as_dict(),
            "score_matched": score_decision.as_dict(),
            "fast_score": fast_score_decision.as_dict(),
            "uncertainty_dispersion": uncertainty_decision.as_dict(),
            "strict_quota": quota_decision.as_dict(),
            "strict_quota_random": quota_random_decision.as_dict(),
            "online_prefix_quota": online_decision.as_dict(),
            "online_prefix_quota_random": online_random_decision.as_dict(),
            "online_route_batch_size": online_batch_size,
            "conformal_local": conformal_decision.as_dict(),
            "conformal_local_random": conformal_random_decision.as_dict(),
            "conformal_threshold": float(conformal_threshold_value),
            "oracle": oracle_decision.as_dict(),
            "calibration": {
                "source": "heldout_normal_only",
                "validation_count": len(validation_records),
                "fallback_budget": float(args.fallback_budget),
            },
        },
        "n_fit": len(fit_records), "n_validation": len(validation_records), "n_test": len(test),
    }
    if gap_alpha:
        gap_threshold = threshold_for_budget(gap_val_scores, args.fallback_budget)
        gap_decision = risk_route(gap_test_scores, gap_threshold)
        gap_random = random_matched_route(len(test), gap_decision.actual_fallback_count, seed=args.seed)
        combined_gap = np.where(gap_decision.mask, full_test_scores, fast_test_scores)
        combined_gap_random = np.where(gap_random.mask, full_test_scores, fast_test_scores)
        result["gap_risk"] = evaluate_mode(
            y_test, combined_gap, gap_decision.actual_rate, fast_ms, full_ms
        )
        result["gap_random_matched"] = evaluate_mode(
            y_test, combined_gap_random, gap_random.actual_rate, fast_ms, full_ms
        )
        result["gap_routing"] = {
            "alpha": gap_alpha,
            "threshold": gap_threshold,
            "risk": gap_decision.as_dict(),
            "random": gap_random.as_dict(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fast-size", type=int, default=128)
    parser.add_argument("--full-size", type=int, default=224)
    parser.add_argument("--fallback-budget", type=float, default=0.25)
    parser.add_argument("--gap-alpha", type=float, default=0.0)
    parser.add_argument("--route-topk", type=int, default=1)
    parser.add_argument("--route-flip", action="store_true")
    parser.add_argument("--route-ms", type=int, default=0)
    parser.add_argument("--route-local", action="store_true")
    parser.add_argument("--local-top-fraction", type=float, default=0.05)
    parser.add_argument("--local-global-alpha", type=float, default=0.0)
    parser.add_argument("--local-flip", action="store_true")
    parser.add_argument("--online-route-batch-size", type=int, default=32)
    parser.add_argument("--full-bank-mode", choices=["all", "image_coreset"], default="all")
    parser.add_argument("--full-bank-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = ResNet18_Weights.DEFAULT
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    fast_transform = transforms.Compose([
        transforms.Resize((args.fast_size, args.fast_size)), transforms.ToTensor(), normalize
    ])
    full_transform = transforms.Compose([
        transforms.Resize((args.full_size, args.full_size)), transforms.ToTensor(), normalize
    ])
    model = ResNetFeatures(args.device)
    index = MVTecADIndex(args.data_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        # The category routine receives the fast transform and creates the
        # full-resolution path explicitly below.
        args._full_transform = full_transform
        result = run_category(index, category, args, model, fast_transform)
        (output / f"{category}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"category": category, **{k: result[k]["image_auroc"] for k in ("fast_only", "full_only", "random_fallback", "risk_fallback")}},), flush=True)


if __name__ == "__main__":
    main()
