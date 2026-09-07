"""Validation-only latency alignment and reproducible bootstrap summaries."""

import numpy as np


def summarize_latency(values):
    """Summarize synchronized per-image latency without assuming CUDA."""
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("latency values must not be empty")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("latency values must be finite and non-negative")
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def select_common_p95_budgets(validation_points, target_p95_ms):
    """Select the largest validation budget not exceeding one shared P95 target."""
    selected = {}
    for method, points in validation_points.items():
        feasible = [point for point in points if point["p95_ms"] <= target_p95_ms]
        if not feasible:
            raise ValueError(f"{method} has no validation budget within target P95")
        selected[method] = max(feasible, key=lambda point: point["budget"])
    return selected


def bootstrap_mean_ci(values, reps=10000, confidence=0.95, seed=17):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(int(reps), values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1.0 - alpha)),
        "reps": int(reps),
        "confidence": float(confidence),
    }


def paired_bootstrap_delta_ci(left, right, reps=10000, confidence=0.95, seed=17):
    left = np.asarray(list(left), dtype=float)
    right = np.asarray(list(right), dtype=float)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired arrays must have equal nonzero length")
    return bootstrap_mean_ci(left - right, reps=reps, confidence=confidence, seed=seed)


def summarize_category_metrics(category_results, metric_names=None, reps=10000, seed=17):
    """Return macro means and paired category bootstrap CIs for JSON results."""
    if not category_results:
        raise ValueError("category_results must not be empty")
    metric_names = metric_names or (
        "image_auroc", "recall_at_fpr", "fallback_rate", "mean_ms", "p95_latency_ms"
    )
    methods = tuple(category_results[0].keys())
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in metric_names:
            values = [result[method][metric] for result in category_results]
            summary[method][metric] = bootstrap_mean_ci(values, reps=reps, seed=seed)
    return summary
