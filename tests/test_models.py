import pytest
from sklearn.metrics import roc_auc_score

from fannie_default import models
from fannie_default.features import out_of_time_split

FACTORIES = ["logistic", "random_forest", "boosting"]


@pytest.mark.parametrize("name", FACTORIES)
def test_model_fits_and_scores(name, loans):
    X_train, X_test, y_train, y_test = out_of_time_split(loans)
    model = getattr(models, name)().fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y_test, p) > 0.7


def test_boosting_early_stopping(loans):
    X_train, X_test, y_train, y_test = out_of_time_split(loans)
    valid = (loans.loc[X_train.index, "vintage"] == loans["vintage"].max() - 4).to_numpy()  # 2016Q4
    model = models.fit_early_stopped(models.boosting(), X_train, y_train, valid)
    n_trees = model[-1].best_iteration_
    assert 0 < n_trees < model[-1].n_estimators
    assert roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) > 0.7
