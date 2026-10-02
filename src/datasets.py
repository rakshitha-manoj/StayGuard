"""Load the two dataset versions as (X, y, meta) for model training.

v1-raw   : data/raw/hotel_bookings.csv (119,390 x 32) made trainable by the MINIMAL
           `prepare_v1_minimal` below, so the comparison shows what real preprocessing adds.
v2-clean : data/processed/hotel_bookings_clean.csv (85,716 x 70), already numeric.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import RAW_FILE, load_raw
from src.utils import DATA_PROCESSED

# Imported lazily-safe: validate.py has no heavy side effects at import time.
from src.validate import LEAKY_COLUMNS, TARGET

CLEAN_FILE = DATA_PROCESSED / "hotel_bookings_clean.csv"
DATA_VERSIONS = ["v1-raw", "v2-clean"]  # also the DVC git tag names


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_v1_minimal(raw: pd.DataFrame) -> pd.DataFrame:
    """Bare-minimum preparation of the raw table so a model can be fit on it.

    Does: drop the 3 leaky columns (LEAKY_COLUMNS; leakage would make v1 meaningless),
    fill nulls naively (numeric -> 0, categorical -> "missing"), one-hot encode every
    object column as-is (country unbucketed, arrival_date_month categorical), keep
    agent/company as raw numeric codes (null -> 0). The target is kept as the last column.

    Does NOT: remove rows, cap outliers (adr, lead_time...), dedupe, or engineer features.

    Caveat: v1 keeps its duplicate rows (31,994 exact duplicates in the raw file; 33,241 once
    the leaky columns are dropped). Copies can land on both sides of the train/test split, which
    may inflate v1 test scores, especially for RF and KNN, which memorise.
    """
    df = raw.drop(columns=LEAKY_COLUMNS).copy()
    obj_cols = sorted(df.select_dtypes(include=["object", "string"]).columns)
    num_cols = [c for c in df.columns if c not in obj_cols]
    df[num_cols] = df[num_cols].fillna(0)
    df[obj_cols] = df[obj_cols].fillna("missing")
    y = df.pop(TARGET)
    df = pd.get_dummies(df, columns=obj_cols, dtype=int)
    df = df.astype(float)
    df[TARGET] = y.astype(int)
    return df


def load_dataset(version: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (X float64 array, y int array, meta). meta has data_version, data_md5,
    feature_names (column order of X), n_rows, positive_rate."""
    if version == "v1-raw":
        path = RAW_FILE
        df = prepare_v1_minimal(load_raw(path))
    elif version == "v2-clean":
        path = CLEAN_FILE
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unknown data version {version!r}; expected one of {DATA_VERSIONS}")
    assert df.columns[-1] == TARGET
    feature_names = list(df.columns[:-1])
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=int)
    meta = {
        "data_version": version,
        "data_md5": file_md5(path),
        "feature_names": feature_names,
        "n_rows": len(df),
        "positive_rate": float(y.mean()),
    }
    return X, y, meta
