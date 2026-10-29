"""Scoring, cost-based decision thresholds and permutation importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def score(y, p) -> dict:
    """Ranking (ROC AUC, PR AUC) and calibration (Brier) metrics, plus the default rate for context.

    PR AUC is the one to watch with rare defaults: its no-skill value is the base
    rate, whereas ROC AUC's is always 0.5.
    """
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "base_rate": float(np.mean(y)),
    }


# Harvey (TX, Aug 2017) and Irma (FL, Sep 2017) forbearance reaches 180+ days
# delinquent for 2016-2017 vintages, an event origination data can't predict.
STORM_STATES = ("FL", "TX")


def score_robust(y, p, states) -> dict:
    """score() on all loans and on loans outside STORM_STATES; a model ranking that
    holds in both is not just learning the hurricanes."""
    y, p, states = np.asarray(y), np.asarray(p), np.asarray(states)
    keep = ~np.isin(states, STORM_STATES)
    return {"all": score(y, p), "excl_storm_states": score(y[keep], p[keep])}


def cost_threshold(y, p, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    """The threshold t (flag loans with p >= t) that minimizes cost_fn*FN + cost_fp*FP.

    Candidates are every distinct score plus +inf (flag nothing). Returns (threshold, total cost).
    """
    y, p = np.asarray(y), np.asarray(p)
    candidates = np.append(np.unique(p), np.inf)
    defaults, good = np.sort(p[y == 1]), np.sort(p[y == 0])
    fn = np.searchsorted(defaults, candidates, side="left")  # defaults scored below t
    fp = len(good) - np.searchsorted(good, candidates, side="left")  # good loans flagged
    cost = cost_fn * fn + cost_fp * fp
    best = int(np.argmin(cost))
    return float(candidates[best]), float(cost[best])


def importance(model, X: pd.DataFrame, y, n_repeats: int = 5) -> pd.Series:
    """Mean drop in ROC AUC when each input column is shuffled, largest first."""
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
