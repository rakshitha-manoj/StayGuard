"""Cleaning and feature engineering: raw bookings -> all-numeric model-ready CSV.

CLI: python -m src.preprocess  (reads params.yaml, writes
data/processed/hotel_bookings_clean.csv and reports_data/preprocess_log.json).
No scaling here; scaling belongs in the model pipeline (Phase 3).
"""
import json

import pandas as pd
import yaml

from src.ingest import RAW_FILE, load_raw
from src.utils import DATA_PROCESSED, PROJECT_ROOT, SEED, set_seed
from src.validate import LEAKY_COLUMNS, REPORTS_DIR, TARGET

PARAMS_FILE = PROJECT_ROOT / "params.yaml"
OUT_FILE = DATA_PROCESSED / "hotel_bookings_clean.csv"
LOG_FILE = REPORTS_DIR / "preprocess_log.json"

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}
CATEGORICALS = ["hotel", "meal", "country", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type"]


def preprocess(df: pd.DataFrame, params: dict, log: dict | None = None) -> pd.DataFrame:
    """Clean `df` per `params` (adr_cap_quantile, top_n_countries, optional seed).

    If `log` is given it is filled with row counts and per-step removals.
    """
    set_seed(params.get("seed", SEED))
    log = log if log is not None else {}
    log["rows_in"] = len(df)
    df = df.copy()

    # 1. Drop target-leaking columns (only known after the outcome; assigned_room_type
    #    is set at check-in, see LEAKY_COLUMNS).
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])

    # 2. Impute: no children recorded -> 0; missing country -> "Unknown".
    log["children_nulls_imputed"] = int(df["children"].isna().sum())
    log["country_nulls_imputed"] = int(df["country"].isna().sum())
    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown")

    # 3. agent / company are high-cardinality nominal ID codes (~333 / ~352 values):
    #    one-hot would explode the width and the numeric ID has no ordinal meaning,
    #    so keep only whether one was involved (null = none) and drop the raw IDs.
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns=["agent", "company"])

    # 4. Features (total_guests is also needed for row filtering below).
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # 5. Remove invalid rows (counted per step).
    n = len(df)
    df = df[df["total_guests"] > 0]
    log["removed_zero_guest_rows"] = n - len(df)
    n = len(df)
    df = df[df["adr"] >= 0]
    log["removed_negative_adr_rows"] = n - len(df)

    # 6. Cap extreme adr at the configured quantile. NOTE: computed on the full
    #    dataset before the train/test split (mild leakage; documented simplification).
    cap = float(df["adr"].quantile(params["adr_cap_quantile"]))
    log["adr_cap_value"] = cap
    log["adr_values_capped"] = int((df["adr"] > cap).sum())
    df["adr"] = df["adr"].clip(upper=cap)

    # 7. Month name -> number 1-12. Year / week / day stay numeric (Phase 4 builds
    #    event_timestamp from them; Phase 6 drift uses later months).
    df["arrival_date_month"] = df["arrival_date_month"].map(MONTHS)

    # 8. Country: keep top-N most frequent, rest -> "Other".
    top = df["country"].value_counts().head(params["top_n_countries"]).index
    log["countries_kept"] = list(top)
    df["country"] = df["country"].where(df["country"].isin(top), "Other")

    # 9. One-hot encode remaining categoricals (0/1 ints), deterministic sorted columns,
    #    target last.
    df = pd.get_dummies(df, columns=CATEGORICALS, dtype=int)
    features = sorted(c for c in df.columns if c != TARGET)
    df = df[features + [TARGET]].reset_index(drop=True)

    # 10. LAST step: drop exact duplicate rows on the final encoded frame (features +
    #     target), keep first, so identical rows cannot land in both train and test and
    #     inflate test metrics (evaluation integrity). Caveat: some duplicates are
    #     legitimate repeated group bookings; we trade a little data for honest metrics.
    n = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    log["removed_duplicate_rows"] = n - len(df)

    log["rows_out"] = len(df)
    log["columns_out"] = df.shape[1]
    return df


def main() -> None:
    params = yaml.safe_load(PARAMS_FILE.read_text())
    cfg = dict(params["preprocess"], seed=params.get("seed", SEED))
    log: dict = {}
    out = preprocess(load_raw(RAW_FILE), cfg, log)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False, lineterminator="\n")  # LF: same md5 on Windows and Linux
    REPORTS_DIR.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(log, indent=2), newline="\n")
    print(json.dumps(log, indent=2))
    print(f"wrote {OUT_FILE} shape={out.shape}")


if __name__ == "__main__":
    main()
