import numpy as np
import pandas as pd

from fannie_default.features import CATEGORICAL, NUMERIC, make_preprocessor, out_of_time_split


def test_split_is_by_vintage(loans):
    X_train, X_test, y_train, y_test = out_of_time_split(loans)
    cut = pd.Period("2017Q1", "Q")
    assert (loans.loc[X_train.index, "vintage"] < cut).all()
    assert (loans.loc[X_test.index, "vintage"] >= cut).all()
    assert len(X_train) + len(X_test) == len(loans)
    assert y_train.index.equals(X_train.index)


def test_no_performance_fields(loans):
    X_train, *_ = out_of_time_split(loans)
    assert set(X_train.columns) <= set(NUMERIC + CATEGORICAL)
    assert "default_24m" not in X_train.columns and "vintage" not in X_train.columns


def test_missing_scores_imputed(loans):
    X_train, *_ = out_of_time_split(loans)
    X_train = X_train.copy()
    X_train.iloc[:50, X_train.columns.get_loc("CSCORE_B")] = np.nan
    X_train.iloc[:50, X_train.columns.get_loc("DTI")] = np.nan
    pre = make_preprocessor(scale=True).fit(X_train)
    out = pre.transform(X_train)
    assert not np.isnan(out).any()
    names = list(pre.get_feature_names_out())
    assert any("missingindicator_CSCORE_B" in n for n in names)
    assert any("missingindicator_DTI" in n for n in names)


def test_unseen_category(loans):
    X_train, X_test, *_ = out_of_time_split(loans)
    X_test = X_test.copy()
    X_test.iloc[0, X_test.columns.get_loc("STATE")] = "GU"
    X_test.iloc[1, X_test.columns.get_loc("STATE")] = np.nan
    pre = make_preprocessor(scale=False).fit(X_train)
    assert pre.transform(X_test).shape[0] == len(X_test)
