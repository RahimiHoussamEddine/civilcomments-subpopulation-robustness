"""Metrics for toxicity prediction and the official overlapping WILDS groups."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.05, 0.951, 0.01), 2))


def group_memberships(
    identities: np.ndarray,
    labels: np.ndarray,
    identity_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Create the 16 overlapping identity-mention × toxicity-label groups."""
    identity_values = np.asarray(identities)
    y = np.asarray(labels, dtype=np.int8)
    if identity_values.ndim != 2 or identity_values.shape != (len(y), len(identity_names)):
        raise ValueError("Identity matrix dimensions do not match labels and names")
    mentioned = identity_values >= 0.5
    columns: list[np.ndarray] = []
    names: list[str] = []
    for index, identity in enumerate(identity_names):
        for label in (0, 1):
            columns.append(mentioned[:, index] & (y == label))
            names.append(f"{identity}|{label}")
    return np.column_stack(columns), names


def group_balanced_weights(memberships: np.ndarray, strength: float) -> np.ndarray:
    """Return bounded inverse-frequency weights for overlapping identity-label groups."""
    groups = np.asarray(memberships, dtype=bool)
    if groups.ndim != 2 or groups.shape[0] == 0 or groups.shape[1] == 0:
        raise ValueError("Group membership matrix must be a non-empty 2D array")
    if strength < 0:
        raise ValueError("Group weight strength cannot be negative")
    group_sizes = groups.sum(axis=0)
    if np.any(group_sizes == 0):
        raise ValueError("Every training group must have at least one example")
    row_factors = groups.astype(np.float64) @ (len(groups) / (groups.shape[1] * group_sizes))
    row_factors = np.minimum(row_factors, 10.0)
    weights = 1.0 + float(strength) * row_factors
    weights /= weights.mean()
    return weights.astype(np.float32)


def _group_accuracy_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    memberships: np.ndarray,
    group_names: Sequence[str],
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    groups = np.asarray(memberships, dtype=bool)
    if groups.shape != (len(y), len(group_names)) or len(pred) != len(y):
        raise ValueError("Group evaluation dimensions do not match predictions")
    correct = (y == pred).astype(np.float64)
    records = []
    for index, name in enumerate(group_names):
        member = groups[:, index]
        count = int(member.sum())
        if count == 0:
            raise ValueError(f"Evaluation group is empty: {name}")
        records.append(
            {
                "group": str(name),
                "count": count,
                "accuracy": float(correct[member].mean()),
            }
        )
    values = [record["accuracy"] for record in records]
    return {
        "accuracy": float(correct.mean()),
        "macro_group_accuracy": float(np.mean(values)),
        "worst_group_accuracy": float(np.min(values)),
        "subgroups": records,
    }


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    y = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if np.any(selected):
            error += float(selected.sum() / total) * abs(
                float(probability[selected].mean()) - float(y[selected].mean())
            )
    return float(error)


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    memberships: np.ndarray,
    group_names: Sequence[str],
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != y.shape or not np.isfinite(probability).all():
        raise ValueError("Probability vector is invalid")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    predictions = (probability >= float(threshold)).astype(np.int8)
    group_summary = _group_accuracy_summary(y, predictions, memberships, group_names)
    return {
        **group_summary,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "expected_calibration_error_15_bins": expected_calibration_error(y, probability),
        "true_positive_rate": float(((predictions == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "true_negative_rate": float(((predictions == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "false_positive_rate": float(
            ((predictions == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
        ),
    }


def threshold_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    memberships: np.ndarray,
    group_names: Sequence[str],
    thresholds: Sequence[float] = THRESHOLDS,
) -> list[dict[str, float]]:
    y = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    points = []
    for threshold in thresholds:
        predictions = (probability >= threshold).astype(np.int8)
        group_summary = _group_accuracy_summary(y, predictions, memberships, group_names)
        points.append(
            {
                "threshold": float(threshold),
                "accuracy": group_summary["accuracy"],
                "macro_group_accuracy": group_summary["macro_group_accuracy"],
                "worst_group_accuracy": group_summary["worst_group_accuracy"],
                "true_positive_rate": float(
                    ((predictions == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
                ),
                "false_positive_rate": float(
                    ((predictions == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
                ),
            }
        )
    return points


def select_threshold(curve: Sequence[dict[str, float]], objective: str) -> float:
    if not curve:
        raise ValueError("Threshold curve cannot be empty")
    if objective not in {"accuracy", "worst_group_accuracy"}:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    best = sorted(
        curve,
        key=lambda row: (
            -row[objective],
            -row["accuracy"],
            round(abs(row["threshold"] - 0.5), 12),
            row["threshold"],
        ),
    )[0]
    return float(best["threshold"])


def article_cluster_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    memberships: np.ndarray,
    article_ids: np.ndarray,
    resamples: int = 500,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    """Percentile intervals for accuracy, resampling source articles as clusters."""
    y = np.asarray(labels, dtype=np.int8)
    pred = np.asarray(predictions, dtype=np.int8)
    groups = np.asarray(memberships, dtype=bool)
    articles = np.asarray(article_ids)
    if len(y) != len(pred) or len(y) != len(articles) or groups.shape[0] != len(y):
        raise ValueError("Bootstrap input dimensions do not match")
    _, cluster_index = np.unique(articles, return_inverse=True)
    cluster_count = int(cluster_index.max()) + 1
    total_by_cluster = np.bincount(cluster_index, minlength=cluster_count).astype(np.float64)
    correct_by_cluster = np.bincount(
        cluster_index, weights=(y == pred).astype(np.float64), minlength=cluster_count
    )
    group_count_by_cluster = np.zeros((cluster_count, groups.shape[1]), dtype=np.float64)
    group_correct_by_cluster = np.zeros_like(group_count_by_cluster)
    correct = (y == pred).astype(np.float64)
    for group_index in range(groups.shape[1]):
        member = groups[:, group_index]
        group_count_by_cluster[:, group_index] = np.bincount(
            cluster_index[member], minlength=cluster_count
        )
        group_correct_by_cluster[:, group_index] = np.bincount(
            cluster_index[member], weights=correct[member], minlength=cluster_count
        )

    rng = np.random.default_rng(seed)
    average_draws = np.empty(resamples, dtype=np.float64)
    worst_group_draws = np.empty(resamples, dtype=np.float64)
    for draw_index in range(resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        multiplicity = np.bincount(sampled, minlength=cluster_count).astype(np.float64)
        average_draws[draw_index] = float(
            (multiplicity @ correct_by_cluster) / (multiplicity @ total_by_cluster)
        )
        group_counts = multiplicity @ group_count_by_cluster
        group_correct = multiplicity @ group_correct_by_cluster
        present = group_counts > 0
        worst_group_draws[draw_index] = float(
            np.min(group_correct[present] / group_counts[present])
        )

    def interval(draws: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(draws, [0.025, 0.975])
        return {"ci_95_lower": float(low), "ci_95_upper": float(high)}

    return {
        "average_accuracy": interval(average_draws),
        "worst_group_accuracy": interval(worst_group_draws),
        "resamples": int(resamples),
        "cluster_unit": "source article",
    }


def paired_article_cluster_bootstrap_delta(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    memberships: np.ndarray,
    article_ids: np.ndarray,
    resamples: int = 500,
    seed: int = 2027,
) -> dict[str, Any]:
    """Paired article-cluster intervals for candidate-minus-baseline accuracy."""
    y = np.asarray(labels, dtype=np.int8)
    base = np.asarray(baseline_predictions, dtype=np.int8)
    candidate = np.asarray(candidate_predictions, dtype=np.int8)
    groups = np.asarray(memberships, dtype=bool)
    articles = np.asarray(article_ids)
    if len(y) != len(base) or len(y) != len(candidate) or len(y) != len(articles):
        raise ValueError("Paired bootstrap input lengths do not match")
    if groups.shape[0] != len(y):
        raise ValueError("Paired bootstrap group rows do not match")
    _, cluster_index = np.unique(articles, return_inverse=True)
    cluster_count = int(cluster_index.max()) + 1
    base_correct = (base == y).astype(np.float64)
    candidate_correct = (candidate == y).astype(np.float64)
    total_by_cluster = np.bincount(cluster_index, minlength=cluster_count).astype(np.float64)
    base_by_cluster = np.bincount(cluster_index, weights=base_correct, minlength=cluster_count)
    candidate_by_cluster = np.bincount(
        cluster_index, weights=candidate_correct, minlength=cluster_count
    )

    group_count = np.zeros((cluster_count, groups.shape[1]), dtype=np.float64)
    base_group_correct = np.zeros_like(group_count)
    candidate_group_correct = np.zeros_like(group_count)
    for group_index in range(groups.shape[1]):
        member = groups[:, group_index]
        group_count[:, group_index] = np.bincount(cluster_index[member], minlength=cluster_count)
        base_group_correct[:, group_index] = np.bincount(
            cluster_index[member], weights=base_correct[member], minlength=cluster_count
        )
        candidate_group_correct[:, group_index] = np.bincount(
            cluster_index[member], weights=candidate_correct[member], minlength=cluster_count
        )

    rng = np.random.default_rng(seed)
    average_delta = np.empty(resamples, dtype=np.float64)
    worst_group_delta = np.empty(resamples, dtype=np.float64)
    for draw_index in range(resamples):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        multiplicity = np.bincount(sampled, minlength=cluster_count).astype(np.float64)
        total = float(multiplicity @ total_by_cluster)
        average_delta[draw_index] = float(
            (multiplicity @ (candidate_by_cluster - base_by_cluster)) / total
        )
        counts = multiplicity @ group_count
        present = counts > 0
        base_group = (multiplicity @ base_group_correct)[present] / counts[present]
        candidate_group = (multiplicity @ candidate_group_correct)[present] / counts[present]
        worst_group_delta[draw_index] = float(candidate_group.min() - base_group.min())

    group_names = [str(index) for index in range(groups.shape[1])]
    base_group_metrics = _group_accuracy_summary(y, base, groups, group_names)
    candidate_group_metrics = _group_accuracy_summary(y, candidate, groups, group_names)

    def summarize(draws: np.ndarray, point: float) -> dict[str, float]:
        low, high = np.quantile(draws, [0.025, 0.975])
        return {"delta": float(point), "ci_95_lower": float(low), "ci_95_upper": float(high)}

    return {
        "candidate_minus_baseline_average_accuracy": summarize(
            average_delta,
            candidate_group_metrics["accuracy"] - base_group_metrics["accuracy"],
        ),
        "candidate_minus_baseline_worst_group_accuracy": summarize(
            worst_group_delta,
            candidate_group_metrics["worst_group_accuracy"]
            - base_group_metrics["worst_group_accuracy"],
        ),
        "resamples": int(resamples),
        "cluster_unit": "source article",
    }
