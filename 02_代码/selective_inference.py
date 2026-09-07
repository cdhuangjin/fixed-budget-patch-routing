"""Risk scoring, threshold selection, and selective Full fallback utilities."""

import math

import torch


class NormalityRiskScorer:
    def __init__(self, eps=1e-6):
        self.eps = float(eps)
        self.mean = None
        self.scale = None

    def fit(self, normal_features):
        features = torch.as_tensor(normal_features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("normal_features must be a non-empty 2D tensor")
        self.mean = features.mean(dim=0)
        self.scale = features.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def score(self, features):
        if self.mean is None or self.scale is None:
            raise RuntimeError("fit must be called before score")
        values = torch.as_tensor(features, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("features must be a 2D tensor with the fitted feature dimension")
        return (((values - self.mean) / self.scale) ** 2).mean(dim=1).sqrt()


def combine_risk(anomaly_score, uncertainty, anomaly_weight=0.7, uncertainty_weight=0.3):
    anomaly_score = torch.as_tensor(anomaly_score, dtype=torch.float32)
    uncertainty = torch.as_tensor(uncertainty, dtype=torch.float32)
    if anomaly_score.shape != uncertainty.shape:
        raise ValueError("anomaly_score and uncertainty must have the same shape")
    anomaly_norm = anomaly_score / anomaly_score.max().clamp_min(1e-6)
    return (anomaly_weight * anomaly_norm + uncertainty_weight * uncertainty).clamp(0.0, 1.0)


def choose_threshold(validation_scores, fallback_budget):
    scores = sorted(float(score) for score in validation_scores)
    if not scores:
        raise ValueError("validation_scores must not be empty")
    if not 0.0 < fallback_budget <= 1.0:
        raise ValueError("fallback_budget must be in (0, 1]")
    high_risk_count = max(1, math.ceil(len(scores) * fallback_budget))
    return scores[-high_risk_count]


def _percentile(values, q):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def selective_predict(
    risk_scores,
    fast_predictions,
    full_predict,
    threshold,
    fast_latency_ms=1.0,
    full_latency_ms=3.0,
):
    scores = list(float(score) for score in risk_scores)
    fallback_indices = [index for index, score in enumerate(scores) if score >= threshold]
    predictions = dict(fast_predictions)
    predictions.update(full_predict(fallback_indices))
    latencies = [full_latency_ms if index in fallback_indices else fast_latency_ms for index in range(len(scores))]
    return {
        "predictions": predictions,
        "fallback": [index in fallback_indices for index in range(len(scores))],
        "fallback_count": len(fallback_indices),
        "fallback_rate": len(fallback_indices) / len(scores) if scores else 0.0,
        "corrected_count": None,
        "new_error_count": None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95) if latencies else 0.0,
    }
