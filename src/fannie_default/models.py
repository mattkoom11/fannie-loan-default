"""Model factories. Each returns an unfitted pipeline that takes the raw X from out_of_time_split."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fannie_default.features import make_preprocessor


def logistic() -> Pipeline:
    # class_weight="balanced" because defaults are well under 1% of loans; scores
    # are then not calibrated probabilities, which the evaluation accounts for.
    return Pipeline([
        ("pre", make_preprocessor(scale=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
