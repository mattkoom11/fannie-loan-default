from pathlib import Path

import pytest

from fannie_default.layout import FIELDS

N_COLUMNS = 113


def write_raw(path: Path, rows: list[dict]) -> Path:
    """Write rows in the headerless, pipe-delimited quarterly format; unset columns are blank."""
    lines = []
    for row in rows:
        cols = [""] * N_COLUMNS
        for name, value in row.items():
            cols[FIELDS[name]] = "" if value is None else str(value)
        lines.append("|".join(cols))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def raw_quarter_file(tmp_path):
    def _write(rows: list[dict], name: str = "2016Q1.csv") -> Path:
        return write_raw(tmp_path / name, rows)

    return _write


def make_loans(n: int = 2_000, seed: int = 0):
    """Loan-level frame shaped like loans.parquet; default risk falls with credit score."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    score = rng.normal(740, 45, n).clip(580, 850)
    p = 1 / (1 + np.exp((score - 640) / 20))
    vintages = pd.period_range("2015Q1", "2017Q4", freq="Q")
    loans = pd.DataFrame({
        "LOAN_ID": [str(100000 + i) for i in range(n)],
        "vintage": pd.PeriodIndex(rng.choice(vintages.astype(str), n), freq="Q"),
        "ORIG_RATE": rng.normal(4.2, 0.4, n),
        "ORIG_UPB": rng.integers(80, 600, n) * 1000.0,
        "ORIG_TERM": rng.choice([180.0, 360.0], n),
        "OLTV": rng.integers(50, 97, n).astype(float),
        "OCLTV": rng.integers(50, 97, n).astype(float),
        "NUM_BO": rng.choice([1.0, 2.0], n),
        "DTI": rng.integers(15, 45, n).astype(float),
        "CSCORE_B": score,
        "CSCORE_C": np.where(rng.random(n) < 0.5, np.nan, score + rng.normal(0, 20, n)),
        "MI_PCT": np.where(rng.random(n) < 0.7, np.nan, 25.0),
        "CHANNEL": rng.choice(["R", "C", "B"], n),
        "FIRST_FLAG": rng.choice(["Y", "N"], n),
        "PURPOSE": rng.choice(["P", "C", "R"], n),
        "PROP": rng.choice(["SF", "PU", "CO"], n),
        "NO_UNITS": rng.choice(["1", "2"], n),
        "OCC_STAT": rng.choice(["P", "S", "I"], n),
        "STATE": rng.choice(["VA", "CA", "TX", "NY"], n),
        "default_24m": (rng.random(n) < p).astype(int),
    })
    return loans


@pytest.fixture
def loans():
    return make_loans()
