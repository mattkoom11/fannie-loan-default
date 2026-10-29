"""Read quarterly performance files in chunks, keeping a deterministic sample of loans."""

import zlib
from pathlib import Path

import pandas as pd

from fannie_default.layout import DATE_FIELDS, FIELDS, NUMERIC_FIELDS

_BUCKETS = 10_000


def in_sample(loan_id: str, rate: float) -> bool:
    # crc32 rather than hash(): Python salts str hashes per process, which would
    # change the sample between runs.
    return zlib.crc32(loan_id.encode()) % _BUCKETS < rate * _BUCKETS


def read_quarter(path: Path, rate: float, chunksize: int = 500_000) -> pd.DataFrame:
    """All rows of the sampled loans in one quarterly file, with typed columns.

    Sampling is by loan, so a kept loan keeps every monthly row even when its
    rows straddle chunk boundaries.
    """
    positions = sorted(FIELDS.values())
    names = {pos: name for name, pos in FIELDS.items()}
    kept = []
    for chunk in pd.read_csv(path, sep="|", header=None, usecols=positions, dtype=str, chunksize=chunksize):
        chunk = chunk.rename(columns=names)
        keep = {i: in_sample(i, rate) for i in chunk["LOAN_ID"].unique()}
        kept.append(chunk[chunk["LOAN_ID"].map(keep)])
    perf = pd.concat(kept, ignore_index=True)

    for col in DATE_FIELDS:
        perf[col] = pd.to_datetime(perf[col], format="%m%Y", errors="coerce").dt.to_period("M")
    for col in NUMERIC_FIELDS:
        perf[col] = pd.to_numeric(perf[col], errors="coerce")
    return perf
