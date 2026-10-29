"""Model factories. Each returns an unfitted pipeline that takes the raw X from out_of_time_split."""

from sklearn.ensemble import RandomForestClassifier
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


def random_forest() -> Pipeline:
    # Trees split on thresholds, so numerics are left unscaled. min_samples_leaf=50 keeps
    # each tree from carving out single defaults; "balanced_subsample" reweights classes
    # within each tree's bootstrap sample. Read OOB results with evaluate.oob_score,
    # since the built-in oob_score_ is accuracy.
    return Pipeline([
        ("pre", make_preprocessor(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )),
    ])
