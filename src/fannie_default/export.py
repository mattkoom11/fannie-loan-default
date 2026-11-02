"""Aggregates-only JSON export of model scores for the portfolio dashboard."""

import numpy as np
import pandas as pd

from fannie_default.evaluate import STORM_STATES, apply_threshold, bayes_threshold
from fannie_default.vintage import default_rate_by

MIN_LOANS = 30  # smaller groups are published without a rate: too noisy, and near-identifying


def cost_ratios(lo: float = 5, hi: float = 200, n: int = 57) -> list[float]:
    """Log-spaced cost_fn:cost_fp ratios from lo to hi, always including 10, 25 and 100."""
    grid = np.round(np.geomspace(lo, hi, n), 4)
    return sorted({float(r) for r in grid} | {10.0, 25.0, 100.0})


def cost_curve(y, p, ratios: list[float]) -> list[dict]:
    """Bayes-threshold outcome at each ratio. `threshold` stays finite even when no
    loan is flagged; that case shows as n_flagged == 0."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for ratio in ratios:
        threshold = bayes_threshold(ratio, 1)
        out = apply_threshold(y, p, threshold, cost_fn=ratio, cost_fp=1)
        flagged = p >= threshold
        rows.append({
            "ratio": ratio,
            "threshold": threshold,
            "flagged": out["flagged"],
            "caught": out["caught"],
            "n_flagged": int(flagged.sum()),
            "n_caught": int((flagged & (y == 1)).sum()),
            "cost": float(out["cost"]),
            "cost_flag_none": float(out["cost_flag_none"]),
            "cost_cut": 1 - out["cost"] / out["cost_flag_none"],
        })
    return rows


def calibration_bins(y, p, n_bins: int = 10, min_loans: int = MIN_LOANS) -> list[dict]:
    """Mean predicted vs observed default rate in quantile bins of the score; bins under min_loans carry no values."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["bin"] = pd.qcut(frame["p"], n_bins, duplicates="drop")
    g = frame.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), observed=("y", "mean"), loans=("y", "size"))
    return [
        {
            "mean_predicted": None if r.loans < min_loans else float(r.mean_predicted),
            "observed": None if r.loans < min_loans else float(r.observed),
            "loans": int(r.loans),
            "suppressed": bool(r.loans < min_loans),
        }
        for r in g.itertuples()
    ]


def add_region(loans: pd.DataFrame) -> pd.DataFrame:
    """Add `region`: the state for STORM_STATES, else "rest of US"."""
    loans = loans.copy()
    loans["region"] = loans["STATE"].where(loans["STATE"].isin(STORM_STATES), "rest of US")
    return loans


def group_rates(loans: pd.DataFrame, by: list[str], min_loans: int = MIN_LOANS) -> list[dict]:
    """Loans, defaults and default rate per group; groups under min_loans carry no rate or defaults."""
    table = default_rate_by(loans, by).reset_index()
    rows = []
    for r in table.to_dict("records"):
        suppressed = r["loans"] < min_loans
        row = {k: str(r[k]) for k in by}
        row["loans"] = int(r["loans"])
        row["defaults"] = None if suppressed else int(r["defaults"])
        row["rate"] = None if suppressed else float(r["default_rate"])
        row["suppressed"] = bool(suppressed)
        rows.append(row)
    return rows
