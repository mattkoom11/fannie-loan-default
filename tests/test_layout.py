import pandas as pd

from fannie_default.layout import FIELDS


def test_fixture_roundtrip(raw_quarter_file):
    path = raw_quarter_file([{"LOAN_ID": "100000000001", "ACT_PERIOD": "012016"}])
    raw = pd.read_csv(path, sep="|", header=None, dtype=str)
    assert raw.iloc[0, FIELDS["LOAN_ID"]] == "100000000001"
    assert raw.iloc[0, FIELDS["ACT_PERIOD"]] == "012016"
