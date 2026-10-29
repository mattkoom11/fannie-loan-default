import pytest
from sklearn.metrics import roc_auc_score

from fannie_default import models
from fannie_default.features import out_of_time_split

# boosting joins this list after learning gate B.
FACTORIES = ["logistic", "random_forest"]


@pytest.mark.parametrize("name", FACTORIES)
def test_model_fits_and_scores(name, loans):
    X_train, X_test, y_train, y_test = out_of_time_split(loans)
    model = getattr(models, name)().fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y_test, p) > 0.7
