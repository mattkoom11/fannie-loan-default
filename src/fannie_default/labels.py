"""Collapse monthly performance rows to one row per loan with a 24-month default label."""

import pandas as pd

from fannie_default.layout import ORIGINATION_FIELDS

# Zero-balance codes for loss events: 02 third-party sale, 03 short sale,
# 09 deed-in-lieu or REO disposition, 15 non-performing note sale.
DEFAULT_ZB_CODES = {"02", "03", "09", "15"}
WINDOW_MONTHS = 24


def _month_number(periods: pd.Series) -> pd.Series:
    return periods.dt.year * 12 + periods.dt.month


def build_loans(perf: pd.DataFrame) -> pd.DataFrame:
    """One row per loan: origination fields, `vintage` and `default_24m`.

    default_24m is 1 if, in the first WINDOW_MONTHS months from the first payment,
    the loan was ever 90+ days delinquent or ended with a loss-type zero-balance
    code. Loans observed for less than the window that neither defaulted nor
    terminated are right-censored and dropped. An unknown delinquency status
    ("XX", blank) counts as missing for that month.
    """
    perf = perf.sort_values(["LOAN_ID", "ACT_PERIOD"])
    month = _month_number(perf["ACT_PERIOD"]) - _month_number(perf["FIRST_PAY"])
    in_window = month.between(0, WINDOW_MONTHS - 1)

    window = pd.DataFrame({
        "LOAN_ID": perf["LOAN_ID"],
        "delinquent_90": pd.to_numeric(perf["DLQ_STATUS"], errors="coerce") >= 3,
        "loss_exit": perf["Zero_Bal_Code"].isin(DEFAULT_ZB_CODES),
        "terminated": perf["Zero_Bal_Code"].notna(),
    })[in_window]
    events = window.groupby("LOAN_ID")[["delinquent_90", "loss_exit", "terminated"]].any()
    last_month = month.groupby(perf["LOAN_ID"]).max()

    loans = perf.drop_duplicates("LOAN_ID", keep="first").set_index("LOAN_ID")[list(ORIGINATION_FIELDS)]
    loans["vintage"] = loans["ORIG_DATE"].dt.asfreq("Q")
    events = events.reindex(loans.index, fill_value=False)
    default = events["delinquent_90"] | events["loss_exit"]
    observed = (last_month.reindex(loans.index) >= WINDOW_MONTHS - 1) | events["terminated"] | default
    loans["default_24m"] = default.astype(int)
    return loans[observed].reset_index()
