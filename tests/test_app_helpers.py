from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from civilcomments_robustness.inference import feature_contributions, probability_for


def test_local_scoring_and_feature_contributions_use_the_supplied_text() -> None:
    texts = ["kind helpful comment", "toxic harmful comment"] * 3
    labels = np.array([0, 1] * 3)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = SGDClassifier(loss="log_loss", random_state=3, max_iter=1000).fit(matrix, labels)
    bundle = {"vectorizer": vectorizer, "erm_model": model}

    probability = probability_for(bundle, "erm_model", "harmful comment")
    contributions = feature_contributions(bundle, "erm_model", "harmful comment")

    assert 0 <= probability <= 1
    assert not contributions.empty
    assert set(contributions.columns) == {"Feature", "Contribution", "Direction"}
