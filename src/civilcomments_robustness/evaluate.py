"""Score validation-selected models once on the filtered, locked WILDS test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from civilcomments_robustness.data import load_leakage_safe_splits
from civilcomments_robustness.metrics import (
    article_cluster_bootstrap,
    evaluate_predictions,
    group_memberships,
    paired_article_cluster_bootstrap_delta,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/civilcomments_models.joblib"
VALIDATION_PATH = ROOT / "reports/validation_metrics.json"
EVALUATION_PATH = ROOT / "reports/evaluation.json"


def _probabilities(model: Any, matrix: Any) -> np.ndarray:
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(matrix)[:, positive_column].astype(np.float64, copy=False)


def evaluate() -> dict[str, Any]:
    if not MODEL_PATH.exists() or not VALIDATION_PATH.exists():
        raise FileNotFoundError("Run `make train` before `make evaluate`")
    splits, exclusions = load_leakage_safe_splits()
    test_frame = splits["test"]
    bundle = joblib.load(MODEL_PATH)
    validation_report = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))

    labels = (test_frame["toxicity"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    identities = test_frame[bundle["identity_columns"]].to_numpy(dtype=np.float32)
    memberships, group_names = group_memberships(identities, labels, bundle["identity_columns"])
    matrix = bundle["vectorizer"].transform(test_frame["comment_text"].fillna("").astype(str))
    erm_probability = _probabilities(bundle["erm_model"], matrix)
    robust_probability = _probabilities(bundle["group_reweighted_model"], matrix)
    erm_threshold = float(bundle["erm_threshold"])
    robust_threshold = float(bundle["group_reweighted_threshold"])
    erm_prediction = (erm_probability >= erm_threshold).astype(np.int8)
    robust_prediction = (robust_probability >= robust_threshold).astype(np.int8)
    article_ids = test_frame["article_id"].to_numpy()

    results: dict[str, Any] = {}
    for name, probability, threshold, prediction in (
        ("erm", erm_probability, erm_threshold, erm_prediction),
        ("group_reweighted", robust_probability, robust_threshold, robust_prediction),
    ):
        selected_metrics = evaluate_predictions(
            labels, probability, threshold, memberships, group_names
        )
        reference_metrics = evaluate_predictions(labels, probability, 0.5, memberships, group_names)
        results[name] = {
            "validation_selected_threshold": threshold,
            "at_validation_selected_threshold": selected_metrics,
            "at_fixed_0_50_threshold": reference_metrics,
            "article_cluster_bootstrap_95ci": article_cluster_bootstrap(
                labels,
                prediction,
                memberships,
                article_ids,
                resamples=500,
                seed=2026 if name == "erm" else 2027,
            ),
        }

    report = {
        "dataset": "WILDS CivilComments v1.0",
        "test_rows": int(len(test_frame)),
        "test_articles": int(test_frame["article_id"].nunique()),
        "toxicity_label_rule": "source toxicity score >= 0.5",
        "group_definition": (
            "One of eight identity-mention annotations >= 0.5 crossed with toxicity "
            "label; groups overlap"
        ),
        "row_exclusions": exclusions,
        "test_labels_used_for_model_or_threshold_selection": False,
        "validation_selection": validation_report["selected"],
        "models": results,
        "paired_group_reweighted_minus_erm_article_cluster_bootstrap_95ci": (
            paired_article_cluster_bootstrap_delta(
                labels,
                erm_prediction,
                robust_prediction,
                memberships,
                article_ids,
                resamples=500,
            )
        ),
        "model_comparison_note": (
            "Both methods use the same train-fitted word TF-IDF features and alpha. "
            "ERM's global threshold was selected for validation average accuracy; the "
            "reweighted model's global threshold and weight strength were selected for "
            "validation worst-group accuracy. The test split was scored after selection."
        ),
        "interpretation_note": (
            "Identity fields indicate mentions in comment text, not author demographics. "
            "Human toxicity labels are subjective crowd annotations. Metrics do not "
            "establish fairness for people or suitability for moderation decisions."
        ),
        "raw_comments_or_row_predictions_committed": False,
    }
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
