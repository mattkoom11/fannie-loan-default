from pathlib import Path

import pytest

from fannie_default.layout import FIELDS

N_COLUMNS = 108


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
