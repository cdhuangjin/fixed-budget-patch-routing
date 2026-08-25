"""Conformal Selective Inference with FPR Control.

Inspired by: CADES - Conformal Anomaly Detection in Event Sequences (ICML 2025)
Key innovation: finite-sample FPR control via conformal inference.

This module implements:
1. Conformal calibration for selective fallback threshold
2. Finite-sample FPR guarantee
3. Group-conditional risk control (per-category)
"""

from __future__ import annotations
import math
from typing import Optional


def conformal_threshold(
    calibration_scores: list[float],
    target_fpr: float = 0.1,
) -> float:
    """Compute conformal threshold for FPR control.
    
    Uses the conformal quantile to guarantee finite-sample FPR control.
    The threshold is set such that the false positive rate on calibration
    data is at most target_fpr.
    
    Args:
        calibration_scores: non-conformity scores from calibration set
        target_fpr: target false positive rate (type-I error)
    
    Returns:
        threshold: conformal threshold for anomaly detection
    """
    if not calibration_scores:
        raise ValueError("calibration_scores must not be empty")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    
    sorted_scores = sorted(calibration_scores)
    n = len(sorted_scores)
    # Conformal quantile: ceil((1-alpha)*(n+1))/n
    quantile_idx = min(n - 1, math.ceil((1 - target_fpr) * (n + 1)) - 1)
    return sorted_scores[max(0, quantile_idx)]


def group_conditional_threshold(
    group_scores: dict[str, list[float]],
    target_fpr: float = 0.1,
) -> dict[str, float]:
    """Compute per-group conformal thresholds.
    
    Ensures FPR control within each group (category),
    which is stronger than marginal FPR control.
    
    Args:
        group_scores: {group_name: [calibration_scores]}
        target_fpr: target FPR per group
    
    Returns:
        {group_name: threshold}
    """
    return {
        group: conformal_threshold(scores, target_fpr)
        for group, scores in group_scores.items()
    }


def select_with_conformal_risk(
    test_scores: list[float],
    fast_scores: list[float],
    full_scores: list[float],
    threshold: float,
    budget: float = 0.25,
) -> dict:
    """Selective inference with conformal risk control.
    
    Samples with non-conformity score >= threshold are "risky"
    and get the full model prediction. Others get fast prediction.
    
    Args:
        test_scores: non-conformity scores for test samples
        fast_scores: fast model predictions
        full_scores: full model predictions
        threshold: conformal threshold
        budget: maximum fallback rate
    
    Returns:
        dict with predictions, fallback_mask, metrics
    """
    n = len(test_scores)
    risky = [i for i, s in enumerate(test_scores) if s >= threshold]
    
    # Apply budget constraint
    max_fallback = max(1, int(n * budget))
    if len(risky) > max_fallback:
        # Keep only the riskiest samples within budget
        risky_sorted = sorted(risky, key=lambda i: -test_scores[i])
        risky = set(risky_sorted[:max_fallback])
    else:
        risky = set(risky)
    
    # Build predictions
    predictions = []
    fallback_mask = []
    for i in range(n):
        if i in risky:
            predictions.append(full_scores[i])
            fallback_mask.append(True)
        else:
            predictions.append(fast_scores[i])
            fallback_mask.append(False)
    
    fallback_rate = sum(fallback_mask) / n
    
    return {
        "predictions": predictions,
        "fallback_mask": fallback_mask,
        "fallback_rate": fallback_rate,
        "risky_count": len(risky),
        "total": n,
        "threshold": threshold,
    }


def compute_fpr(
    labels: list[int],
    predictions: list[float],
    threshold: float,
) -> float:
    """Compute false positive rate at given threshold."""
    negatives = [p for l, p in zip(labels, predictions) if l == 0]
    if not negatives:
        return 0.0
    return sum(1 for p in negatives if p >= threshold) / len(negatives)


def compute_power(
    labels: list[int],
    predictions: list[float],
    threshold: float,
) -> float:
    """Compute statistical power (true positive rate) at given threshold."""
    positives = [p for l, p in zip(labels, predictions) if l == 1]
    if not positives:
        return 0.0
    return sum(1 for p in positives if p >= threshold) / len(positives)
