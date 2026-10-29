import random

import pandas as pd

from fannie_default.build import quarter_files
from fannie_default.labels import build_loans
from fannie_default.load import read_quarter


def history(loan_id, n_months, dlq=None, zb=None, cscore=720, first_pay="2016-01"):
    """Monthly rows for one loan. Month k is k-1 months after the first payment.

    dlq: {month: status}, default "0". zb: (month, code), ending the loan there.
    """
    dlq = dlq or {}
    start = pd.Period(first_pay, "M")
    rows = []
    for k in range(1, n_months + 1):
        rows.append({
            "LOAN_ID": loan_id,
            "ACT_PERIOD": (start + (k - 1)).strftime("%m%Y"),
            "ORIG_DATE": (start - 2).strftime("%m%Y"),
            "FIRST_PAY": start.strftime("%m%Y"),
            "CSCORE_B": cscore,
            "DLQ_STATUS": dlq.get(k, "0"),
        })
        if zb and k == zb[0]:
            rows[-1]["Zero_Bal_Code"] = zb[1]
            break
    return rows


def label(raw_quarter_file, rows):
    loans = build_loans(read_quarter(raw_quarter_file(rows), rate=1.0))
    return loans.set_index("LOAN_ID")


def test_90_day_delinquency_is_default(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 36, dlq={8: "1", 9: "2", 10: "3"}))
    assert loans.loc["1", "default_24m"] == 1


def test_late_delinquency_outside_window(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 36, dlq={30: "3"}))
    assert loans.loc["1", "default_24m"] == 0


def test_foreclosure_code_is_default(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 40, zb=(15, "09")))
    assert loans.loc["1", "default_24m"] == 1


def test_prepaid_is_not_default(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 40, zb=(8, "01")))
    assert loans.loc["1", "default_24m"] == 0


def test_unknown_status_is_missing(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 24, dlq={3: "XX", 4: None}))
    assert loans.loc["1", "default_24m"] == 0


def test_censored_loan_dropped(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 24) + history("2", 12))
    assert "1" in loans.index
    assert "2" not in loans.index


def test_features_from_first_record(raw_quarter_file):
    rows = history("1", 24, cscore=650)
    rows[0]["CSCORE_B"] = 720
    random.Random(0).shuffle(rows)
    loans = label(raw_quarter_file, rows)
    assert loans.loc["1", "CSCORE_B"] == 720


def test_one_row_per_loan_with_vintage(raw_quarter_file):
    loans = label(raw_quarter_file, history("1", 24) + history("2", 30, first_pay="2016-03"))
    assert len(loans) == 2
    assert loans.loc["1", "vintage"] == pd.Period("2015Q4", "Q")
    assert loans.loc["2", "vintage"] == pd.Period("2016Q1", "Q")
    assert "DLQ_STATUS" not in loans.columns


def test_quarter_files(tmp_path):
    files = quarter_files(tmp_path, "2015Q3", "2016Q2")
    assert [f.name for f in files] == ["2015Q3.csv", "2015Q4.csv", "2016Q1.csv", "2016Q2.csv"]
