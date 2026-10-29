"""Model inputs: which origination fields, how they are preprocessed, and the out-of-time split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fannie_default.layout import NUMERIC_FIELDS

NUMERIC = list(NUMERIC_FIELDS)
CATEGORICAL = ["CHANNEL", "FIRST_FLAG", "PURPOSE", "PROP", "NO_UNITS", "OCC_STAT", "STATE"]


def out_of_time_split(loans: pd.DataFrame, test_start: str = "2017Q1"):
    """Train on vintages before `test_start`, test on the rest. Returns X_train, X_test, y_train, y_test."""
    is_test = loans["vintage"] >= pd.Period(test_start, "Q")
    X = loans[NUMERIC + CATEGORICAL]
    y = loans["default_24m"]
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Median-impute numerics with missing indicators (scaled for linear models); one-hot categoricals.

    Rare categories (under 0.5% of training rows) are pooled into one "infrequent"
    column, and categories first seen at prediction time go to that column too.
    """
    numeric = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        numeric.append(StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False),
    )
    return ColumnTransformer([
        ("num", make_pipeline(*numeric), NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
