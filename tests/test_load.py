import pandas as pd

from fannie_default.load import in_sample, read_quarter


def test_in_sample_deterministic():
    ids = [f"{100000000000 + i}" for i in range(200)]
    assert [in_sample(i, 0.3) for i in ids] == [in_sample(i, 0.3) for i in ids]
    assert all(in_sample(i, 1.0) for i in ids)
    assert not any(in_sample(i, 0.0) for i in ids)


def test_in_sample_rate():
    ids = [f"{100000000000 + i}" for i in range(20_000)]
    share = sum(in_sample(i, 0.1) for i in ids) / len(ids)
    assert 0.09 <= share <= 0.11


def test_dates_keep_leading_zero(raw_quarter_file):
    path = raw_quarter_file([{"LOAN_ID": "1", "ACT_PERIOD": "012016", "ORIG_DATE": "112015", "FIRST_PAY": "012016"}])
    perf = read_quarter(path, rate=1.0)
    assert perf.loc[0, "ACT_PERIOD"] == pd.Period("2016-01", "M")
    assert perf.loc[0, "ORIG_DATE"] == pd.Period("2015-11", "M")


def test_numeric_blank_is_nan_and_status_stays_text(raw_quarter_file):
    path = raw_quarter_file([{"LOAN_ID": "1", "ACT_PERIOD": "012016", "CSCORE_B": None, "DTI": "36", "DLQ_STATUS": "XX"}])
    perf = read_quarter(path, rate=1.0)
    assert pd.isna(perf.loc[0, "CSCORE_B"])
    assert perf.loc[0, "DTI"] == 36.0
    assert perf.loc[0, "DLQ_STATUS"] == "XX"


def test_sampled_loan_keeps_all_rows(raw_quarter_file):
    months = pd.period_range("2016-01", periods=30, freq="M")
    rows = [{"LOAN_ID": "777", "ACT_PERIOD": m.strftime("%m%Y")} for m in months]
    perf = read_quarter(raw_quarter_file(rows), rate=1.0, chunksize=7)
    assert len(perf) == 30
