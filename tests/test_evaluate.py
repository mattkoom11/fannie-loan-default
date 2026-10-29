import numpy as np
import pytest

from fannie_default import models
from fannie_default.evaluate import cost_threshold, importance, score
from fannie_default.features import out_of_time_split

Y = np.array([0, 0, 1, 0, 1, 1])
P = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])


def test_cost_threshold_toy():
    # Hand-solved: flagging p >= 0.3 catches every default with one false alarm.
    threshold, cost = cost_threshold(Y, P, cost_fn=2, cost_fp=1)
    assert threshold == pytest.approx(0.3)
    assert cost == pytest.approx(1)


def test_cost_threshold_extremes():
    # The lowest-scored loan defaults, so catching every default means flagging everyone.
    y_low_default = np.array([1, 0, 0, 1, 0, 1])
    flag_all, _ = cost_threshold(y_low_default, P, cost_fn=1000, cost_fp=1)
    assert flag_all == pytest.approx(P.min())
    # The highest-scored loan is good, so any flag costs a false alarm.
    y_top_good = np.array([1, 1, 0, 1, 1, 0])
    flag_none, cost = cost_threshold(y_top_good, P, cost_fn=1, cost_fp=1000)
    assert flag_none > P.max()
    assert cost == pytest.approx(y_top_good.sum())


def test_score_keys():
    result = score(Y, P)
    assert set(result) == {"roc_auc", "pr_auc", "brier", "base_rate"}
    assert result["base_rate"] == pytest.approx(0.5)
    assert 0 <= result["roc_auc"] <= 1


def test_importance_ranks_credit_score_first(loans):
    X_train, X_test, y_train, y_test = out_of_time_split(loans)
    model = models.logistic().fit(X_train, y_train)
    imp = importance(model, X_test, y_test, n_repeats=3)
    assert imp.index[0] == "CSCORE_B"
    assert set(imp.index) == set(X_test.columns)
    assert imp.is_monotonic_decreasing
