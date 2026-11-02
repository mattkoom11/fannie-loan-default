import json

import numpy as np
import pytest

from fannie_default.export import add_region, calibration_bins, cost_curve, cost_ratios, group_rates


def test_cost_ratios_hits_the_marked_ratios():
    r = cost_ratios()
    assert r == sorted(set(r)) and r[0] == 5 and r[-1] == 200
    assert {10.0, 25.0, 100.0} <= set(r)


def test_cost_curve_matches_apply_threshold():
    y = np.array([0, 0, 1, 0, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    (pt,) = cost_curve(y, p, [9.0])  # threshold 0.1: flags everyone
    assert pt["threshold"] == pytest.approx(0.1)
    assert pt["n_flagged"] == 6 and pt["n_caught"] == 3 and pt["caught"] == 1.0
    assert pt["cost_cut"] == pytest.approx(1 - pt["cost"] / pt["cost_flag_none"])


def test_cost_curve_flag_nothing_is_null_not_infinity():
    y = np.array([0, 1]); p = np.array([0.001, 0.002])
    (pt,) = cost_curve(y, p, [5.0])
    assert pt["n_flagged"] == 0 and pt["flagged"] == 0
    json.dumps(pt, allow_nan=False)  # raises on inf/nan


def test_calibration_bins_cover_every_loan(loans):
    y = loans["default_24m"].to_numpy(); p = np.linspace(0, 1, len(y))
    bins = calibration_bins(y, p)
    assert len(bins) == 10 and sum(b["loans"] for b in bins) == len(y)


def test_group_rates_suppresses_small_cells(loans):
    rows = group_rates(add_region(loans), ["vintage", "region"], min_loans=10**9)
    assert rows and all(r["suppressed"] and r["rate"] is None and r["defaults"] is None for r in rows)
    json.dumps(rows, allow_nan=False)


def test_group_rates_keeps_large_cells(loans):
    rows = group_rates(add_region(loans), ["region"], min_loans=1)
    assert {r["region"] for r in rows} == {"TX", "rest of US"}
    assert sum(r["loans"] for r in rows) == len(loans)
