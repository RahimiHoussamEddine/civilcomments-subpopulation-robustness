"""Fit and validation-select CPU sparse toxicity models without using test labels."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from civilcomments_robustness.data import load_leakage_safe_splits
from civilcomments_robustness.metrics import (
    evaluate_predictions,
    group_balanced_weights,
    group_memberships,
    select_threshold,
    threshold_curve,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
MODEL_PATH = ROOT / "models/civilcomments_models.joblib"
VALIDATION_PATH = ROOT / "reports/validation_metrics.json"
ALPHA_CANDIDATES = (1e-5, 3e-5, 1e-4)
GROUP_STRENGTH_CANDIDATES = (0.5, 1.0, 2.0)
MAX_FEATURES = 80_000
MAX_ITER = 20
SEED = 2026


def _labels(frame: Any) -> np.ndarray:
    return (frame["toxicity"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)


def _new_classifier(alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=MAX_ITER,
        tol=1e-3,
        shuffle=True,
        random_state=SEED,
        average=True,
    )


def _positive_probability(model: SGDClassifier, matrix: Any) -> np.ndarray:
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(matrix)[:, positive_column].astype(np.float64, copy=False)


def train() -> dict[str, Any]:
    started = time.perf_counter()
    splits, exclusions = load_leakage_safe_splits()
    train_frame = splits["train"]
    validation_frame = splits["val"]
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    train_labels = _labels(train_frame)
    validation_labels = _labels(validation_frame)
    identity_names = list(manifest["identity_columns"])
    train_group_matrix, _ = group_memberships(
        train_frame[identity_names].to_numpy(dtype=np.float32), train_labels, identity_names
    )
    validation_groups, group_names = group_memberships(
        validation_frame[identity_names].to_numpy(dtype=np.float32),
        validation_labels,
        identity_names,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.995,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_frame["comment_text"].fillna("").astype(str))
    validation_matrix = vectorizer.transform(
        validation_frame["comment_text"].fillna("").astype(str)
    )

    erm_candidates: list[dict[str, Any]] = []
    selected_erm: SGDClassifier | None = None
    selected_erm_probability: np.ndarray | None = None
    best_log_loss = float("inf")
    for alpha in ALPHA_CANDIDATES:
        model = _new_classifier(alpha)
        model.fit(train_matrix, train_labels)
        probability = _positive_probability(model, validation_matrix)
        value = float(log_loss(validation_labels, probability, labels=[0, 1]))
        erm_candidates.append(
            {
                "alpha": alpha,
                "validation_log_loss": value,
                "n_iter": int(model.n_iter_),
            }
        )
        if value < best_log_loss:
            best_log_loss = value
            selected_erm = model
            selected_erm_probability = probability

    if selected_erm is None or selected_erm_probability is None:
        raise RuntimeError("No ERM model was selected")
    selected_alpha = float(selected_erm.alpha)
    erm_curve = threshold_curve(
        validation_labels, selected_erm_probability, validation_groups, group_names
    )
    erm_threshold = select_threshold(erm_curve, "accuracy")
    erm_validation_metrics = evaluate_predictions(
        validation_labels,
        selected_erm_probability,
        erm_threshold,
        validation_groups,
        group_names,
    )

    robust_candidates: list[dict[str, Any]] = []
    selected_robust: SGDClassifier | None = None
    selected_robust_probability: np.ndarray | None = None
    selected_robust_curve: list[dict[str, float]] | None = None
    selected_robust_threshold: float | None = None
    selected_robust_strength: float | None = None
    selected_robust_metrics: dict[str, Any] | None = None
    best_robust_key: tuple[float, float, float] | None = None

    for strength in GROUP_STRENGTH_CANDIDATES:
        sample_weights = group_balanced_weights(train_group_matrix, strength)
        model = _new_classifier(selected_alpha)
        model.fit(train_matrix, train_labels, sample_weight=sample_weights)
        probability = _positive_probability(model, validation_matrix)
        curve = threshold_curve(validation_labels, probability, validation_groups, group_names)
        threshold = select_threshold(curve, "worst_group_accuracy")
        metrics = evaluate_predictions(
            validation_labels, probability, threshold, validation_groups, group_names
        )
        key = (
            float(metrics["worst_group_accuracy"]),
            float(metrics["accuracy"]),
            -abs(threshold - 0.5),
        )
        robust_candidates.append(
            {
                "group_weight_strength": strength,
                "threshold": threshold,
                "validation_metrics": metrics,
                "n_iter": int(model.n_iter_),
            }
        )
        if best_robust_key is None or key > best_robust_key:
            best_robust_key = key
            selected_robust = model
            selected_robust_probability = probability
            selected_robust_curve = curve
            selected_robust_threshold = threshold
            selected_robust_strength = float(strength)
            selected_robust_metrics = metrics

    if (
        selected_robust is None
        or selected_robust_probability is None
        or selected_robust_curve is None
        or selected_robust_threshold is None
        or selected_robust_strength is None
        or selected_robust_metrics is None
    ):
        raise RuntimeError("No group-reweighted model was selected")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer": vectorizer,
        "erm_model": selected_erm,
        "group_reweighted_model": selected_robust,
        "identity_columns": identity_names,
        "selected_alpha": selected_alpha,
        "selected_group_weight_strength": selected_robust_strength,
        "erm_threshold": erm_threshold,
        "group_reweighted_threshold": selected_robust_threshold,
        "feature_count": len(vectorizer.vocabulary_),
        "max_features": MAX_FEATURES,
        "max_iter": MAX_ITER,
        "seed": SEED,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    validation_report = {
        "dataset": manifest["dataset"],
        "validation_rows_used": int(len(validation_frame)),
        "train_rows_used": int(len(train_frame)),
        "row_exclusions": exclusions,
        "test_data_used_for_fitting_or_model_selection": False,
        "test_text_used_only_for_predeclared_exact_duplicate_exclusion": True,
        "feature_method": "word TF-IDF, unigram and bigram, max 80,000 terms",
        "estimator": "SGDClassifier logistic loss, L2 penalty, averaged weights",
        "max_iter": MAX_ITER,
        "erm_alpha_candidates": erm_candidates,
        "group_weight_strength_candidates": robust_candidates,
        "selected": {
            "erm": {
                "alpha": selected_alpha,
                "threshold_objective": "validation average accuracy",
                "threshold": erm_threshold,
                "validation_metrics": erm_validation_metrics,
            },
            "group_reweighted": {
                "alpha": selected_alpha,
                "group_weight_strength": selected_robust_strength,
                "threshold_objective": "validation worst-group accuracy",
                "threshold": selected_robust_threshold,
                "validation_metrics": selected_robust_metrics,
            },
        },
        "threshold_curves": {
            "erm": erm_curve,
            "group_reweighted": selected_robust_curve,
        },
        "features": int(len(vectorizer.vocabulary_)),
        "train_matrix_nnz": int(train_matrix.nnz),
        "training_seconds": float(time.perf_counter() - started),
        "model_artifact_committed": False,
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation_report, indent=2))
    return validation_report


def main() -> None:
    train()


if __name__ == "__main__":
    main()
