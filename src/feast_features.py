"""Feast helper CLI: build the feature source, run the training/inference demo.

    python -m src.feast_features prepare         # CSV -> feature_repo/data/*.parquet
    python -m src.feast_features demo            # historical vs online fetch + consistency check
    python -m src.feast_features training-frame  # point-in-time join for all bookings

Between prepare and demo run `feast apply` and `feast materialize` in feature_repo/.
"""
import argparse
import io
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from src.utils import DATA_PROCESSED, FEATURE_REPO, PROJECT_ROOT

CLEAN_CSV = DATA_PROCESSED / "hotel_bookings_clean.csv"
FEATURES_PARQUET = FEATURE_REPO / "data" / "booking_features.parquet"
LABELS_PARQUET = FEATURE_REPO / "data" / "booking_labels.parquet"
DEMO_OUTPUT = PROJECT_ROOT / "reports_data" / "feast_demo.txt"

FEATURES = ["lead_time", "adr", "previous_cancellations"]
FEATURE_REFS = [f"booking_features:{f}" for f in FEATURES]
SAMPLE_IDS = [0, 1000, 20000, 50000, 85715]
CHECK_ID = 20000


def prepare() -> None:
    """Write the feature source (3 features) and a separate label table.

    booking_id is the row position in the v2-clean CSV (0..n-1), so it is stable only
    for a given data version (md5 0bd1d207...). event_timestamp is the arrival date
    (UTC midnight). created_timestamp equals event_timestamp so the output is
    deterministic; it only breaks ties between duplicate (id, event_timestamp) rows.
    is_canceled is the label, not a feature: it goes to a separate parquet and joins in
    through the entity dataframe at training time.
    """
    df = pd.read_csv(CLEAN_CSV)
    ts = pd.to_datetime(
        {"year": df["arrival_date_year"], "month": df["arrival_date_month"],
         "day": df["arrival_date_day_of_month"]}, utc=True)
    ids = np.arange(len(df), dtype="int64")
    feats = pd.DataFrame({
        "booking_id": ids,
        "event_timestamp": ts,
        "created_timestamp": ts,
        "lead_time": df["lead_time"].astype("int64"),
        "adr": df["adr"].astype("float32"),
        "previous_cancellations": df["previous_cancellations"].astype("int64"),
    })
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES_PARQUET, index=False)
    labels = pd.DataFrame({"booking_id": ids, "event_timestamp": ts,
                           "is_canceled": df["is_canceled"].astype("int64")})
    labels.to_parquet(LABELS_PARQUET, index=False)
    print(f"wrote {FEATURES_PARQUET} {feats.shape} and {LABELS_PARQUET} {labels.shape}")
    print(f"event_timestamp range: {ts.min()} .. {ts.max()}")


def _store():
    from feast import FeatureStore
    return FeatureStore(repo_path=str(FEATURE_REPO))


def demo() -> bool:
    store = _store()
    labels = pd.read_parquet(LABELS_PARQUET)
    raw = pd.read_csv(CLEAN_CSV)

    print("=== Shared definition ===")
    fv = store.get_feature_view("booking_features")
    print("FeatureView:", fv.name, "| entities:", fv.entities, "| ttl:", fv.ttl,
          "| online:", fv.online)
    print("features:", [f"{f.name}:{f.dtype}" for f in fv.features])
    fs = store.get_feature_service("stayguard_model_v1")
    print("FeatureService:", fs.name, "-> views:",
          [p.name for p in fs.feature_view_projections])
    print("Feature refs used by BOTH calls below:", FEATURE_REFS)

    print("\n=== 1. Historical (training) fetch, point-in-time correct ===")
    entity_df = labels[labels["booking_id"].isin(SAMPLE_IDS)].reset_index(drop=True)
    hist = store.get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()
    print(hist.to_string(index=False))

    print(f"\n=== 1b. Point-in-time check: same 5 bookings, booking {CHECK_ID} "
          "requested 1 day BEFORE its event ===")
    early = entity_df.copy()
    early.loc[early["booking_id"] == CHECK_ID, "event_timestamp"] -= pd.Timedelta(days=1)
    hist_early = store.get_historical_features(entity_df=early, features=FEATURE_REFS).to_df()
    print(hist_early.to_string(index=False))
    # CHECK_ID's only feature row is stamped one day AFTER the requested time, so it
    # must not be used. Feast 0.66's file (dask) offline store left-joins entity rows to
    # feature rows on the key only, then _filter_ttl keeps candidates with
    # feature_ts <= entity_ts (and the null-feature rows of unknown keys). A known key
    # whose every candidate is in the future therefore loses its entity row: no null row.
    leaked = hist_early[hist_early["booking_id"] == CHECK_ID]
    others_same = hist_early.sort_values("booking_id").reset_index(drop=True).equals(
        hist[hist["booking_id"] != CHECK_ID].sort_values("booking_id").reset_index(drop=True))
    pit_ok = bool((len(leaked) == 0 or leaked[FEATURES].isna().all(axis=None)) and others_same)
    print(f"rows requested: {len(early)}, returned: {len(hist_early)}; rows for booking "
          f"{CHECK_ID}: {len(leaked)} (a leak would show lead_time=105 etc.); "
          f"other 4 bookings unchanged: {others_same}")

    print("\n=== 1c. Null example: booking_id that has no feature row at all ===")
    unseen = pd.DataFrame({"booking_id": [10_000_000],
                           "event_timestamp": [pd.Timestamp("2017-09-01", tz="UTC")],
                           "is_canceled": [0]})
    hist_unseen = store.get_historical_features(entity_df=unseen, features=FEATURE_REFS).to_df()
    print(hist_unseen.to_string(index=False))
    print("PIT check:", "PASS (no future value leaked into the past)" if pit_ok else "FAIL")

    print("\n=== 2. Online (inference) fetch for one booking ===")
    # to_dict() key order varies between runs, so print in a fixed order.
    keys = ["booking_id"] + FEATURES
    resp = store.get_online_features(
        features=FEATURE_REFS, entity_rows=[{"booking_id": CHECK_ID}]).to_dict()
    online = {k: resp[k] for k in keys}
    print(online)
    via_fs = store.get_online_features(
        features=fs, entity_rows=[{"booking_id": CHECK_ID}]).to_dict()
    print("Same call via the FeatureService:", {k: via_fs[k] for k in keys})

    print(f"\n=== 3. Consistency check for booking_id={CHECK_ID} ===")
    h = hist[hist["booking_id"] == CHECK_ID].iloc[0]
    r = raw.iloc[CHECK_ID]
    ok = True
    for f in FEATURES:
        o, hv, rv = online[f][0], h[f].item(), r[f].item()
        # Online and historical must match exactly. The CSV is float64 while the view
        # stores adr as Float32 (~7 significant digits), so CSV is compared at rtol=1e-6.
        same = bool(float(o) == float(hv)
                    and np.isclose(float(hv), float(rv), rtol=1e-6, atol=0))
        ok &= same
        print(f"{f:24s} online={o!r:>10} historical={hv!r:>10} csv={rv!r:>10} "
              f"{'OK' if same else 'MISMATCH'}")
    ok &= pit_ok
    print("CONSISTENCY CHECK:", "PASS" if ok else "FAIL")
    return ok


def training_frame() -> None:
    store = _store()
    labels = pd.read_parquet(LABELS_PARQUET)
    t = time.perf_counter()
    df = store.get_historical_features(entity_df=labels, features=FEATURE_REFS).to_df()
    dt = time.perf_counter() - t
    print(f"point-in-time join for all bookings: {len(df):,} rows x {df.shape[1]} cols in {dt:.1f}s")
    print("null feature cells:", int(df[FEATURES].isna().sum().sum()))
    print(df.head().to_string(index=False))


class _Tee(io.TextIOBase):
    def __init__(self, buf):
        self.buf = buf

    def write(self, s):
        self.buf.write(s)
        return sys.__stdout__.write(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["prepare", "demo", "training-frame"])
    cmd = p.parse_args().command
    if cmd == "prepare":
        prepare()
    elif cmd == "training-frame":
        training_frame()
    else:
        buf = io.StringIO()
        with redirect_stdout(_Tee(buf)):
            ok = demo()
        DEMO_OUTPUT.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
