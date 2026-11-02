import json
from pathlib import Path

import numpy as np
import pytest

from fannie_default.export import (
    add_region,
    build_dashboard,
    calibration_bins,
    check_reproduction,
    cost_curve,
    cost_ratios,
    group_rates,
    main,
    seed_from_metrics,
)


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


def test_cost_curve_flag_nothing_stays_finite():
    y = np.array([0, 1]); p = np.array([0.001, 0.002])
    (pt,) = cost_curve(y, p, [5.0])
    assert pt["n_flagged"] == 0 and pt["flagged"] == 0
    assert pt["threshold"] == pytest.approx(1 / 6)
    json.dumps(pt, allow_nan=False)  # raises on inf/nan


def test_calibration_bins_cover_every_loan(loans):
    y = loans["default_24m"].to_numpy(); p = np.linspace(0, 1, len(y))
    bins = calibration_bins(y, p)
    assert len(bins) == 10 and sum(b["loans"] for b in bins) == len(y)


def test_calibration_bins_suppress_small_bins(loans):
    y = loans["default_24m"].to_numpy(); p = np.linspace(0, 1, len(y))
    bins = calibration_bins(y, p, min_loans=10**9)
    assert all(b["suppressed"] and b["mean_predicted"] is None and b["observed"] is None and b["loans"] > 0 for b in bins)
    json.dumps(bins, allow_nan=False)
    assert not any(b["suppressed"] for b in calibration_bins(y, p))


def test_group_rates_suppresses_small_cells(loans):
    rows = group_rates(add_region(loans), ["vintage", "region"], min_loans=10**9)
    assert rows and all(r["suppressed"] and r["rate"] is None and r["defaults"] is None for r in rows)
    json.dumps(rows, allow_nan=False)


def test_group_rates_keeps_large_cells(loans):
    rows = group_rates(add_region(loans), ["region"], min_loans=1)
    assert {r["region"] for r in rows} == {"TX", "rest of US"}
    assert sum(r["loans"] for r in rows) == len(loans)


def test_build_dashboard_shape_and_valid_json(loans):
    dash = build_dashboard(loans, git_sha="abc123")
    assert set(dash) == {"summary", "metrics", "random_forest_oob", "cost_curve", "calibration",
                         "vintage_region", "fico_ltv", "importance", "meta"}
    assert set(dash["metrics"]) == {"logistic", "random_forest", "lightgbm"}
    assert any(pt["ratio"] == 100.0 for pt in dash["cost_curve"])
    json.dumps(dash, allow_nan=False)


def test_build_dashboard_has_no_loan_ids(loans):
    text = json.dumps(build_dashboard(loans, git_sha="x"))
    assert not any(lid in text for lid in loans["LOAN_ID"].head(50))


def test_seed_from_real_metrics():
    metrics = json.loads(Path("reports/metrics.json").read_text())
    seed = seed_from_metrics(metrics, git_sha="x")
    assert [pt["ratio"] for pt in seed["cost_curve"]] == [10.0, 25.0, 100.0]
    hundred = seed["cost_curve"][2]
    assert hundred["flagged"] == pytest.approx(0.0689, abs=1e-4) and hundred["caught"] == pytest.approx(0.3927, abs=1e-4)
    assert seed["calibration"] is None and seed["meta"]["source"] == "seed"
    json.dumps(seed, allow_nan=False)


def test_check_reproduction_flags_drift():
    metrics = json.loads(Path("reports/metrics.json").read_text())
    dash = seed_from_metrics(metrics, git_sha="x")
    assert check_reproduction(dash, metrics) == []
    dash["metrics"]["lightgbm"]["all"]["roc_auc"] += 0.003
    assert any("lightgbm" in m for m in check_reproduction(dash, metrics))


def test_check_reproduction_flags_brier_drift():
    metrics = json.loads(Path("reports/metrics.json").read_text())
    dash = seed_from_metrics(metrics, git_sha="x")
    dash["metrics"]["random_forest"]["all"]["brier"] += 0.001
    msgs = check_reproduction(dash, metrics)
    assert len(msgs) == 1 and "random_forest" in msgs[0] and "Brier" in msgs[0]


def test_main_fails_and_writes_nothing_on_drift(tmp_path, monkeypatch, capsys):
    real = json.loads(Path("reports/metrics.json").read_text())
    doctored = json.loads(json.dumps(real))
    for by_subset in doctored["metrics"].values():
        for ref in by_subset.values():
            ref["roc_auc"] += 0.01
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(doctored))
    loans_path = tmp_path / "loans.parquet"
    loans_path.write_bytes(b"")
    monkeypatch.setattr("fannie_default.export.pd.read_parquet", lambda _p: None)
    monkeypatch.setattr("fannie_default.export.build_dashboard", lambda _df, _sha: seed_from_metrics(real, "x"))
    monkeypatch.setattr("fannie_default.export._git_sha", lambda: "x")
    out = tmp_path / "dash.json"
    code = main(["--out", str(out), "--metrics", str(metrics_path), "--loans", str(loans_path)])
    assert code == 1
    assert not out.exists()
    assert "ROC AUC" in capsys.readouterr().err
