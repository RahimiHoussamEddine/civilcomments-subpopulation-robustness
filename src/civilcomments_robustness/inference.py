"""Local predictions and model-specific term contributions for sparse linear models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def probability_for(bundle: dict[str, Any], model_name: str, text: str) -> float:
    """Return the classifier's estimated positive-class probability for one input."""
    vector = bundle["vectorizer"].transform([text])
    model = bundle[model_name]
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return float(model.predict_proba(vector)[0, positive_column])


def feature_contributions(
    bundle: dict[str, Any], model_name: str, text: str, limit: int = 8
) -> pd.DataFrame:
    """Return the highest absolute TF-IDF × linear-coefficient contributions."""
    vector = bundle["vectorizer"].transform([text])
    model = bundle[model_name]
    names = bundle["vectorizer"].get_feature_names_out()
    products = vector.multiply(model.coef_[0]).tocoo()
    if not products.nnz:
        return pd.DataFrame(columns=["Feature", "Contribution", "Direction"])
    rows = [
        (str(names[index]), float(value), "Raises score" if value > 0 else "Lowers score")
        for index, value in zip(products.col, products.data, strict=True)
    ]
    rows.sort(key=lambda row: abs(row[1]), reverse=True)
    return pd.DataFrame(rows[:limit], columns=["Feature", "Contribution", "Direction"])
