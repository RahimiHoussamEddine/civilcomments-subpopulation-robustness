import numpy as np

from civilcomments_robustness.metrics import (
    article_cluster_bootstrap,
    evaluate_predictions,
    expected_calibration_error,
    group_balanced_weights,
    group_memberships,
    select_threshold,
    threshold_curve,
)


def test_group_memberships_preserve_overlapping_identity_mentions() -> None:
    identities = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    labels = np.array([1, 0, 0, 1])
    groups, names = group_memberships(identities, labels, ["alpha", "beta"])

    assert groups.shape == (4, 4)
    assert names == ["alpha|0", "alpha|1", "beta|0", "beta|1"]
    assert groups[0].tolist() == [False, True, False, True]


def test_group_balanced_weights_are_positive_and_mean_normalized() -> None:
    memberships = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=bool)
    weights = group_balanced_weights(memberships, strength=1.0)

    assert np.all(weights > 0)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] > weights[2]


def test_group_metrics_and_validation_threshold_selection() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.2, 0.8, 0.6, 0.4])
    memberships = np.eye(4, dtype=bool)
    names = ["g0", "g1", "g2", "g3"]
    curve = threshold_curve(labels, probabilities, memberships, names, thresholds=(0.3, 0.5, 0.7))
    selected = select_threshold(curve, "accuracy")
    result = evaluate_predictions(labels, probabilities, selected, memberships, names)

    assert selected == 0.3
    assert result["accuracy"] == 0.75
    assert result["worst_group_accuracy"] == 0.0
    assert len(result["subgroups"]) == 4


def test_threshold_tie_break_is_stable_and_prefers_nearest_half() -> None:
    curve = threshold_curve(
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.8, 0.6, 0.4]),
        np.eye(4, dtype=bool),
        ["g0", "g1", "g2", "g3"],
        thresholds=(0.3, 0.5, 0.7),
    )

    assert select_threshold(curve, "accuracy") == 0.3


def test_expected_calibration_error_for_perfectly_calibrated_pairs() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.0, 1.0, 0.0, 1.0])

    assert expected_calibration_error(labels, probabilities) == 0.0


def test_article_cluster_bootstrap_is_reproducible() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1, 0, 0])
    memberships = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=bool)
    articles = np.array([10, 10, 20, 20, 30, 30])
    first = article_cluster_bootstrap(
        labels, predictions, memberships, articles, resamples=100, seed=7
    )
    second = article_cluster_bootstrap(
        labels, predictions, memberships, articles, resamples=100, seed=7
    )

    assert first == second
    assert first["average_accuracy"]["ci_95_lower"] <= 4 / 6
