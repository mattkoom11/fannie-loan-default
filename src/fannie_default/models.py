"""Model factories. Each returns an unfitted pipeline that takes the raw X from out_of_time_split."""

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
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


def boosting() -> Pipeline:
    # Many shallow trees, each fit to the current model's errors and added at a 0.03
    # learning rate. Row (subsample) and column (colsample_bytree) sampling decorrelate
    # the trees like a forest's bootstrap does. n_estimators is a ceiling; fit with
    # fit_early_stopped to pick the actual number of trees.
    return Pipeline([
        ("pre", make_preprocessor(scale=False)),
        ("clf", LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            metric="auc",
            random_state=42,
            verbose=-1,
        )),
    ])


def fit_early_stopped(model: Pipeline, X, y, valid: np.ndarray, patience: int = 100) -> Pipeline:
    """Fit a boosting pipeline on the rows outside `valid`, stopping once validation
    AUC has not improved for `patience` trees; predictions then use the best tree count.

    The validation rows are held out of both the preprocessor and the trees.
    """
    valid = np.asarray(valid, dtype=bool)
    y = np.asarray(y)
    pre, clf = model["pre"], model[-1]
    X_fit = pre.fit_transform(X[~valid])
    X_valid = pre.transform(X[valid])
    clf.fit(
        X_fit, y[~valid],
        eval_X=X_valid, eval_y=y[valid],
        callbacks=[lightgbm.early_stopping(patience, verbose=False)],
    )
    return model
