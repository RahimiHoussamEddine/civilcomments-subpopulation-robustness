import pandas as pd

from civilcomments_robustness.data import leakage_safe_splits


def test_leakage_safe_splits_remove_empty_and_cross_split_duplicate_text() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "val", "val", "test", "test"],
            "comment_text": [
                "Repeat this text",
                "train only",
                "  ",
                "repeat THIS text",
                "same test text",
                "REPEAT this text",
                "same test text",
            ],
            "toxicity": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "article_id": [1, 1, 1, 2, 2, 3, 4],
        }
    )

    splits, exclusions = leakage_safe_splits(frame)

    assert splits["train"]["comment_text"].tolist() == ["train only"]
    assert splits["val"]["comment_text"].tolist() == []
    assert splits["test"]["comment_text"].tolist() == [
        "REPEAT this text",
        "same test text",
    ]
    assert exclusions == {
        "train_removed_for_empty_or_duplicate_text": 2,
        "val_removed_for_empty_or_duplicate_text": 2,
        "test_removed_for_empty_or_duplicate_text": 0,
    }
