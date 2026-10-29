"""Column positions in the Single-Family Loan Performance combined quarterly files.

The files are pipe-delimited with no header row. Only the fields this project
uses are listed, by 0-based column position.

UNVERIFIED: positions were written from memory of Fannie Mae's file layout.
Check every one against crt-file-layout-and-glossary.xlsx before trusting a
real-data build, and record the glossary version here.
"""

FIELDS: dict[str, int] = {
    "LOAN_ID": 1,
    "ACT_PERIOD": 2,
    "CHANNEL": 3,
    "ORIG_RATE": 7,
    "ORIG_UPB": 9,
    "ORIG_TERM": 12,
    "ORIG_DATE": 13,
    "FIRST_PAY": 14,
    "OLTV": 19,
    "OCLTV": 20,
    "NUM_BO": 21,
    "DTI": 22,
    "CSCORE_B": 23,
    "CSCORE_C": 24,
    "FIRST_FLAG": 25,
    "PURPOSE": 26,
    "PROP": 27,
    "NO_UNITS": 28,
    "OCC_STAT": 29,
    "STATE": 30,
    "MI_PCT": 33,
    "DLQ_STATUS": 39,
    "Zero_Bal_Code": 43,
}

# MMYYYY strings; read as text so the leading zero of the month survives.
DATE_FIELDS = ("ACT_PERIOD", "ORIG_DATE", "FIRST_PAY")

ORIGINATION_FIELDS: tuple[str, ...] = tuple(
    name for name in FIELDS if name not in ("LOAN_ID", "ACT_PERIOD", "DLQ_STATUS", "Zero_Bal_Code")
)

NUMERIC_FIELDS: tuple[str, ...] = (
    "ORIG_RATE", "ORIG_UPB", "ORIG_TERM", "OLTV", "OCLTV",
    "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "MI_PCT",
)
