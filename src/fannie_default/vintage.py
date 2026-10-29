"""Default-rate tables by origination vintage and borrower risk bands."""

import numpy as np
import pandas as pd

FICO_LABELS = ["<620", "620-679", "680-739", "740+"]
LTV_LABELS = ["<=80", "80-90", "90-95", ">95"]


def add_bands(loans: pd.DataFrame) -> pd.DataFrame:
    """Add `fico_band` (borrower credit score) and `ltv_band` (original LTV) as ordered categories."""
    loans = loans.copy()
    fico = pd.cut(loans["CSCORE_B"], [-np.inf, 620, 680, 740, np.inf], right=False, labels=FICO_LABELS)
    loans["fico_band"] = fico.cat.add_categories("missing").fillna("missing")
    loans["ltv_band"] = pd.cut(loans["OLTV"], [-np.inf, 80, 90, 95, np.inf], labels=LTV_LABELS)
    return loans


def default_rate_by(loans: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    table = loans.groupby(by, observed=True)["default_24m"].agg(loans="size", defaults="sum")
    table["default_rate"] = table["defaults"] / table["loans"]
    return table
