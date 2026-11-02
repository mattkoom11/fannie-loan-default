"""Aggregates-only JSON export of model scores for the portfolio dashboard."""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fannie_default import models
from fannie_default.evaluate import STORM_STATES, apply_threshold, bayes_threshold, importance, oob_score, score_robust
from fannie_default.features import out_of_time_split
from fannie_default.vintage import add_bands, default_rate_by

SEED = 42
AUC_TOLERANCE = 0.002  # per-model test ROC AUC vs reports/metrics.json
RATE_TOLERANCE = 0.005  # flagged / caught share at 100:1 vs reports/metrics.json
VALIDATION_VINTAGE = "2016Q4"  # LightGBM early-stopping slice, as in notebook 02

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
            "cost": round(float(out["cost"]), 4),
            "cost_flag_none": round(float(out["cost_flag_none"]), 4),
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


def _floats(d: dict) -> dict:
    return {k: float(v) for k, v in d.items()}


def build_dashboard(loans: pd.DataFrame, git_sha: str) -> dict:
    """Refit the three models (notebook 02), score the 2017 test set and return aggregates only."""
    np.random.seed(SEED)
    loans = add_region(add_bands(loans))
    X_train, X_test, y_train, y_test = out_of_time_split(loans)
    valid = (loans.loc[X_train.index, "vintage"] == pd.Period(VALIDATION_VINTAGE, "Q")).to_numpy()
    fitted = {
        "logistic": models.logistic().fit(X_train, y_train),
        "random_forest": models.random_forest().fit(X_train, y_train),
        "lightgbm": models.fit_early_stopped(models.boosting(), X_train, y_train, valid),
    }
    p = {name: m.predict_proba(X_test)[:, 1] for name, m in fitted.items()}
    imp = importance(fitted["lightgbm"], X_test, y_test, n_repeats=5)
    fico_ltv = loans[loans["fico_band"] != "missing"]  # bands are vintage.FICO_LABELS x LTV_LABELS
    return {
        "summary": {
            "loans": len(loans),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_default_rate": float(y_train.mean()),
            "test_default_rate": float(y_test.mean()),
            "overall_default_rate": float(loans["default_24m"].mean()),
        },
        "metrics": {
            name: {subset: _floats(m) for subset, m in score_robust(y_test, p[name], X_test["STATE"]).items()}
            for name in fitted
        },
        "random_forest_oob": _floats(oob_score(fitted["random_forest"], y_train)),
        "cost_curve": cost_curve(y_test, p["lightgbm"], cost_ratios()),
        "calibration": {name: calibration_bins(y_test, p[name]) for name in fitted},
        "vintage_region": group_rates(loans, ["vintage", "region"]),
        "fico_ltv": group_rates(fico_ltv, ["fico_band", "ltv_band"]),
        "importance": [{"feature": f, "auc_drop": float(v)} for f, v in imp.items()],
        "meta": _meta(git_sha, "full"),
    }


def _meta(git_sha: str, source: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "sample_rate": 0.10,
        "min_loans": MIN_LOANS,
        "source": source,
    }


def seed_from_metrics(metrics: dict, git_sha: str) -> dict:
    """Dashboard-shaped dict from the committed reports/metrics.json alone: cost points and
    model metrics only; the loan-level sections are None."""
    n_train, n_test = metrics["n_train"], metrics["n_test"]
    test_rate = metrics["metrics"]["lightgbm"]["all"]["base_rate"]
    n_defaults = round(test_rate * n_test)
    curve = []
    for row in metrics["thresholds_lightgbm_bayes"]:
        ratio = float(row["cost_fn:cost_fp"].split(":")[0])
        threshold = float(row["threshold"])
        curve.append({
            "ratio": ratio,
            "threshold": threshold if math.isfinite(threshold) else None,
            "flagged": float(row["flagged"]),
            "caught": float(row["caught"]),
            "n_flagged": round(row["flagged"] * n_test),
            "n_caught": round(row["caught"] * n_defaults),
            "cost": float(row["cost"]),
            "cost_flag_none": float(row["cost_flag_none"]),
            "cost_cut": float(row["cost cut vs flag none"]),
        })
    imp = sorted(metrics["lightgbm_importance"].items(), key=lambda kv: kv[1], reverse=True)
    return {
        "summary": {
            "loans": n_train + n_test,
            "n_train": n_train,
            "n_test": n_test,
            "train_default_rate": None,
            "test_default_rate": test_rate,
            "overall_default_rate": None,
        },
        "metrics": {
            name: {subset: _floats(m) for subset, m in by_subset.items()}
            for name, by_subset in metrics["metrics"].items()
        },
        "random_forest_oob": _floats(metrics["random_forest_oob"]),
        "cost_curve": curve,
        "calibration": None,
        "vintage_region": None,
        "fico_ltv": None,
        "importance": [{"feature": f, "auc_drop": float(v)} for f, v in imp],
        "meta": _meta(git_sha, "seed"),
    }


def check_reproduction(dash: dict, metrics: dict) -> list[str]:
    """One message per breach of the tolerances against reports/metrics.json; empty means pass."""
    problems = []
    for name, by_subset in metrics["metrics"].items():
        for subset, ref in by_subset.items():
            got = dash["metrics"][name][subset]["roc_auc"]
            if abs(got - ref["roc_auc"]) > AUC_TOLERANCE:
                problems.append(f"{name} {subset} ROC AUC {got:.4f} vs reference {ref['roc_auc']:.4f} (tolerance {AUC_TOLERANCE})")
    ref = next(r for r in metrics["thresholds_lightgbm_bayes"] if r["cost_fn:cost_fp"] == "100:1")
    got = next(pt for pt in dash["cost_curve"] if pt["ratio"] == 100.0)
    for key in ("flagged", "caught"):
        if abs(got[key] - ref[key]) > RATE_TOLERANCE:
            problems.append(f"lightgbm 100:1 {key} {got[key]:.4f} vs reference {ref[key]:.4f} (tolerance {RATE_TOLERANCE})")
    return problems


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loans", type=Path, default=Path("data/processed/loans.parquet"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--seed", action="store_true", help="write the seed from metrics.json instead of refitting")
    args = parser.parse_args(argv)

    metrics = json.loads(args.metrics.read_text())
    if args.seed:
        dash = seed_from_metrics(metrics, _git_sha())
    else:
        dash = build_dashboard(pd.read_parquet(args.loans), _git_sha())
        problems = check_reproduction(dash, metrics)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
    args.out.write_text(json.dumps(dash, indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
