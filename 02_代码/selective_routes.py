"""Deterministic, label-free routing policies for the 027 MVTec protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteDecision:
    mask: np.ndarray
    route_name: str
    threshold: float | None
    target_fallback_count: int
    route_source: str

    @property
    def actual_fallback_count(self) -> int:
        return int(self.mask.sum())

    @property
    def actual_rate(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0

    def as_dict(self) -> dict:
        return {
            "route_name": self.route_name,
            "threshold": self.threshold,
            "target_fallback_count": self.target_fallback_count,
            "actual_fallback_count": self.actual_fallback_count,
            "actual_rate": self.actual_rate,
            "route_source": self.route_source,
        }


def _validate_scores(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def _validate_count(n: int, fallback_count: int) -> int:
    n = int(n)
    fallback_count = int(fallback_count)
    if n < 0 or fallback_count < 0 or fallback_count > n:
        raise ValueError("fallback_count must be between 0 and n")
    return fallback_count


def risk_route(scores, threshold: float) -> RouteDecision:
    values = _validate_scores(scores)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return RouteDecision(
        mask=values >= threshold,
        route_name="risk",
        threshold=threshold,
        target_fallback_count=int(np.count_nonzero(values >= threshold)),
        route_source="validation_calibrated_score",
    )


def random_matched_route(n: int, fallback_count: int, seed: int = 17) -> RouteDecision:
    fallback_count = _validate_count(n, fallback_count)
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(int(n), dtype=bool)
    if fallback_count:
        mask[rng.choice(int(n), size=fallback_count, replace=False)] = True
    return RouteDecision(
        mask=mask,
        route_name="random_matched",
        threshold=None,
        target_fallback_count=fallback_count,
        route_source="seeded_uniform_sample_without_replacement",
    )


def score_matched_route(scores, fallback_count: int, route_name: str = "score_matched") -> RouteDecision:
    values = _validate_scores(scores)
    fallback_count = _validate_count(len(values), fallback_count)
    mask = np.zeros(len(values), dtype=bool)
    if fallback_count:
        # Stable ordering makes ties reproducible and prevents hidden label use.
        selected = np.argsort(-values, kind="mergesort")[:fallback_count]
        mask[selected] = True
    return RouteDecision(
        mask=mask,
        route_name=route_name,
        threshold=None,
        target_fallback_count=fallback_count,
        route_source="test-free_score_ranking",
    )


def quota_route(scores, fallback_budget: float, route_name: str = "strict_quota") -> RouteDecision:
    """Select exactly the largest-risk samples under a declared test quota.

    This is a transductive evaluation route: the batch size is known and the
    quota is enforced directly, so distribution shift cannot inflate the
    fallback rate. It is intentionally reported separately from the
    calibration-only route and must not be described as an online guarantee.
    """
    values = _validate_scores(scores)
    budget = float(fallback_budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("fallback_budget must be between 0 and 1")
    count = int(np.ceil(len(values) * budget))
    mask = np.zeros(len(values), dtype=bool)
    if count:
        selected = np.argsort(-values, kind="mergesort")[:count]
        mask[selected] = True
    return RouteDecision(
        mask=mask,
        route_name=route_name,
        threshold=None,
        target_fallback_count=count,
        route_source="test_free_exact_quota_score_ranking",
    )


def strict_quota_combined_scores(scores, fallback_budget: float, seed: int, boost: float):
    """Build a label-free strict-quota route and an equal-budget random control."""
    values = _validate_scores(scores)
    boost = float(boost)
    if not np.isfinite(boost) or boost <= 0.0:
        raise ValueError("boost must be finite and positive")
    risk = quota_route(values, fallback_budget, route_name="strict_quota")
    control = random_matched_route(
        len(values), risk.actual_fallback_count, seed=seed
    )
    return (
        np.where(risk.mask, values * boost, values),
        np.where(control.mask, values * boost, values),
        risk,
        control,
    )


def online_prefix_quota_route(
    scores, fallback_budget: float, batch_size: int, route_name: str = "online_prefix_quota"
) -> RouteDecision:
    """Apply an exact cumulative budget using only the current score batch.

    Within each arriving batch, only the current cheap-route scores are used.
    The number escalated in a batch is the increment of the cumulative quota,
    so the total count is exactly ``ceil(n * budget)`` without inspecting
    future scores. This is an operational, batch-streaming route rather than a
    transductive global ranking.
    """
    values = _validate_scores(scores)
    budget = float(fallback_budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("fallback_budget must be between 0 and 1")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mask = np.zeros(len(values), dtype=bool)
    selected_so_far = 0
    for start in range(0, len(values), batch_size):
        end = min(len(values), start + batch_size)
        target_end = int(np.ceil(end * budget))
        take = target_end - selected_so_far
        take = max(0, min(end - start, take))
        if take:
            order = np.argsort(-values[start:end], kind="mergesort")[:take]
            mask[start + order] = True
        selected_so_far += take
    return RouteDecision(
        mask=mask,
        route_name=route_name,
        threshold=None,
        target_fallback_count=int(np.ceil(len(values) * budget)),
        route_source="online_batch_prefix_exact_quota_score_ranking",
    )


def oracle_matched_route(labels, fallback_count: int) -> RouteDecision:
    """Upper bound only; never use this route for a primary claim."""
    values = np.asarray(labels, dtype=int).reshape(-1)
    if values.size == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("labels must be a non-empty binary array")
    fallback_count = _validate_count(len(values), fallback_count)
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:fallback_count]] = True
    return RouteDecision(
        mask=mask,
        route_name="oracle_upper_bound",
        threshold=None,
        target_fallback_count=fallback_count,
        route_source="test_labels_upper_bound_only",
    )


def threshold_for_budget(validation_scores, fallback_budget: float) -> float:
    values = _validate_scores(validation_scores)
    budget = float(fallback_budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("fallback_budget must be between 0 and 1")
    return float(np.quantile(values, 1.0 - budget, method="linear"))


def conformal_threshold(validation_scores, fallback_budget: float) -> float:
    """Finite-sample upper conformal quantile from normal calibration scores."""
    values = np.sort(_validate_scores(validation_scores))
    budget = float(fallback_budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("fallback_budget must be between 0 and 1")
    rank = int(np.ceil((len(values) + 1) * (1.0 - budget))) - 1
    rank = min(len(values) - 1, max(0, rank))
    return float(values[rank])
