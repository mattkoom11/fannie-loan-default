"""Build data/processed/loans.parquet from the raw quarterly files.

    uv run python -m fannie_default.build --raw data/raw --quarters 2015Q1 2017Q4 --rate 0.10 \
        --out data/processed/loans.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from fannie_default.labels import build_loans
from fannie_default.load import read_quarter


def quarter_files(raw: Path, start: str, end: str) -> list[Path]:
    return [raw / f"{q.year}Q{q.quarter}.csv" for q in pd.period_range(start, end, freq="Q")]


def in_vintages(loans: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep loans originated in [start, end]. Acquisition files also hold a few seasoned
    loans originated earlier, and the last vintage is partial because some of its loans
    are acquired in the following quarter."""
    vintage = loans["vintage"]
    return loans[(vintage >= pd.Period(start, "Q")) & (vintage <= pd.Period(end, "Q"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--quarters", nargs=2, default=["2015Q1", "2017Q4"], metavar=("START", "END"))
    parser.add_argument("--rate", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=Path("data/processed/loans.parquet"))
    args = parser.parse_args()

    # Each file holds the loans acquired in that quarter, so a loan's rows never
    # span files and each file can be labeled on its own.
    parts = []
    for path in quarter_files(args.raw, *args.quarters):
        loans = build_loans(read_quarter(path, args.rate))
        print(f"{path.name}: {len(loans):,} loans, default rate {loans['default_24m'].mean():.3%}")
        parts.append(loans)
    loans = in_vintages(pd.concat(parts, ignore_index=True), *args.quarters)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(args.out, index=False)
    print(f"total: {len(loans):,} loans, default rate {loans['default_24m'].mean():.3%} -> {args.out}")


if __name__ == "__main__":
    main()
