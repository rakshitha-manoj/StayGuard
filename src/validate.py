"""Data-quality checks for the raw and cleaned bookings datasets.

Raw stage: HARD checks raise `DataValidationError`; SOFT checks are only counted
(the preprocessing step is what fixes them).
Clean stage: everything is STRICT, any violation raises.

CLI: python -m src.validate --stage raw|clean  -> writes reports_data/validation_<stage>.json
and exits non-zero on failure.
"""
import argparse
import json
import sys

import pandas as pd

from src.ingest import EXPECTED_COLUMNS, RAW_FILE, load_raw
from src.utils import DATA_PROCESSED, PROJECT_ROOT

TARGET = "is_canceled"
# Known target leakage. reservation_status(_date) are only known after the booking
# outcome. assigned_room_type is set at check-in: where it differs from the reserved
# type (14,917 rows) the cancel rate is 5.4% vs 41.6% when equal, so it leaks the label.
LEAKY_COLUMNS = ["reservation_status", "reservation_status_date", "assigned_room_type"]
ADR_CAP_QUANTILE = 0.995  # same default as params.yaml preprocess.adr_cap_quantile
REPORTS_DIR = PROJECT_ROOT / "reports_data"
CLEAN_FILE = DATA_PROCESSED / "hotel_bookings_clean.csv"


class DataValidationError(Exception):
    """Raised when a dataset violates a hard validation rule."""


def _target_failures(df: pd.DataFrame) -> list[str]:
    """Failures related to the target column (exists, no nulls, binary)."""
    if TARGET not in df.columns:
        return [f"target column '{TARGET}' is missing"]
    failures = []
    n_null = int(df[TARGET].isna().sum())
    if n_null:
        failures.append(f"target '{TARGET}' has {n_null} null values")
    values = set(df[TARGET].dropna().unique())
    if not values <= {0, 1}:
        failures.append(f"target '{TARGET}' is not binary {{0,1}}: found {sorted(values)}")
    return failures


def _dtype_failures(df: pd.DataFrame) -> list[str]:
    """Columns whose dtype kind contradicts EXPECTED_COLUMNS. Int and float count
    as the same kind (a numeric column gains NaNs and becomes float)."""
    wrong = []
    for col, kind in EXPECTED_COLUMNS.items():
        if col not in df.columns:
            continue
        actual = df[col].dtype.kind
        ok = actual in "iuf" if kind in "if" else (actual == "O" or pd.api.types.is_string_dtype(df[col]))
        if not ok:
            wrong.append(f"{col} (expected kind '{kind}', got '{actual}')")
    return [f"wrong column dtypes: {wrong}"] if wrong else []


def validate_raw(df: pd.DataFrame) -> dict:
    """Validate the raw dataframe. Raises DataValidationError on hard failures,
    otherwise returns a report that includes soft-issue counts."""
    failures = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        failures.append(f"missing expected columns: {missing}")
    failures += _dtype_failures(df)
    if len(df) == 0:
        failures.append("dataframe is empty")
    failures += _target_failures(df)
    if failures:
        raise DataValidationError("Raw data failed validation:\n - " + "\n - ".join(failures))

    nulls = df.isna().sum()
    adr_cap = float(df["adr"].quantile(ADR_CAP_QUANTILE))
    total_guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    soft = {
        "nulls_per_column": {c: int(n) for c, n in nulls.items() if n > 0},
        "negative_adr_rows": int((df["adr"] < 0).sum()),
        "adr_above_cap_rows": int((df["adr"] > adr_cap).sum()),
        "adr_cap_value": adr_cap,
        "zero_guest_rows": int((total_guests == 0).sum()),
        "duplicate_rows": int(df.duplicated().sum()),  # reported, deliberately NOT dropped
        "leaky_columns_present": [c for c in LEAKY_COLUMNS if c in df.columns],
    }
    return {
        "stage": "raw",
        "passed": True,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "soft_issues": soft,
    }


def validate_clean(df: pd.DataFrame) -> dict:
    """Strictly validate the cleaned dataframe; raises on any violation."""
    failures = []
    n_nulls = int(df.isna().sum().sum())
    if n_nulls:
        failures.append(f"{n_nulls} null values present")
    # also catches one-hot leftovers such as assigned_room_type_A
    leaky = [c for c in df.columns
             if c in LEAKY_COLUMNS or any(c.startswith(f"{l}_") for l in LEAKY_COLUMNS)]
    if leaky:
        failures.append(f"leaky columns present: {leaky}")
    if "adr" in df.columns and (df["adr"] < 0).any():
        failures.append(f"{int((df['adr'] < 0).sum())} rows with negative adr")
    if "total_guests" in df.columns:
        if (df["total_guests"] == 0).any():
            failures.append(f"{int((df['total_guests'] == 0).sum())} rows with zero total_guests")
    else:
        failures.append("column 'total_guests' is missing")
    # Evaluation integrity: identical rows must not straddle a train/test split.
    n_dup = int(df.duplicated().sum())
    if n_dup:
        failures.append(f"{n_dup} duplicate rows present")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        failures.append(f"non-numeric columns: {non_numeric}")
    failures += _target_failures(df)
    if failures:
        raise DataValidationError("Clean data failed validation:\n - " + "\n - ".join(failures))
    return {
        "stage": "clean",
        "passed": True,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "nulls": 0,
        "duplicate_rows": 0,
        "cancel_rate": float(df[TARGET].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["raw", "clean"], required=True)
    args = parser.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"validation_{args.stage}.json"
    try:
        if args.stage == "raw":
            report = validate_raw(load_raw(RAW_FILE))
        else:
            report = validate_clean(pd.read_csv(CLEAN_FILE))
    except DataValidationError as e:
        out.write_text(json.dumps({"stage": args.stage, "passed": False, "error": str(e)}, indent=2), newline="\n")
        print(e, file=sys.stderr)
        return 1
    out.write_text(json.dumps(report, indent=2), newline="\n")  # LF on every OS
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
