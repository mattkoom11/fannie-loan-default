import numpy as np
import pandas as pd

from fannie_default.vintage import add_bands, default_rate_by


def toy():
    return pd.DataFrame({
        "vintage": pd.PeriodIndex(["2015Q1"] * 4 + ["2015Q2"] * 2 + ["2015Q3"] * 4, freq="Q"),
        "CSCORE_B": [600, 650, 700, 750, 760, np.nan, 610, 619, 620, 740],
        "OLTV": [80, 81, 90, 91, 95, 96, 60, 85, 92, 99],
        "default_24m": [1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    })


def test_rate_by_vintage():
    table = default_rate_by(toy(), "vintage")
    assert list(table.columns) == ["loans", "defaults", "default_rate"]
    assert table["loans"].tolist() == [4, 2, 4]
    assert table["defaults"].tolist() == [1, 1, 2]
    assert table["default_rate"].tolist() == [0.25, 0.5, 0.5]


def test_rate_by_two_keys():
    table = default_rate_by(add_bands(toy()), ["vintage", "fico_band"])
    assert isinstance(table.index, pd.MultiIndex)
    assert table.loc[(pd.Period("2015Q3", "Q"), "<620"), "loans"] == 2
    assert table.loc[(pd.Period("2015Q3", "Q"), "<620"), "default_rate"] == 1.0


def test_band_edges():
    banded = add_bands(toy())
    assert banded["fico_band"].astype(str).tolist()[6:10] == ["<620", "<620", "620-679", "740+"]
    assert banded["ltv_band"].astype(str).tolist()[:6] == ["<=80", "80-90", "80-90", "90-95", "90-95", ">95"]


def test_missing_fico_band():
    banded = add_bands(toy())
    assert banded.loc[5, "fico_band"] == "missing"
    assert "missing" in default_rate_by(banded, "fico_band").index
