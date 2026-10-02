# Phase 4: Feast feature store

Feast 0.66.0, local Parquet offline store, SQLite online store. All outputs below were captured from real runs on Windows 11 (Python 3.11). The full demo output is also saved in [reports_data/feast_demo.txt](../reports_data/feast_demo.txt).

## 1. What is in the repo

| Piece | Where | Definition |
|---|---|---|
| Entity | `feature_repo/features.py` | `booking_id`, INT64 join key. Synthetic: the row position (0..85,715) in the v2-clean CSV. |
| Data source | `feature_repo/features.py` | `FileSource` on `data/booking_features.parquet`, `timestamp_field=event_timestamp`, `created_timestamp_column=created_timestamp`. |
| FeatureView | `feature_repo/features.py` | `booking_features`: exactly three features, `lead_time` (Int64), `adr` (Float32), `previous_cancellations` (Int64); `online=True`. |
| FeatureService | `feature_repo/features.py` | `stayguard_model_v1` groups the view; it is what a model version points at. |
| Store config | `feature_repo/feature_store.yaml` | project `stayguard`, provider `local`, registry `data/registry.db`, online store SQLite `data/online_store.db`, offline store `file`, `entity_key_serialization_version: 3` (the current recommended value; lower values trigger a deprecation warning). |
| Helper CLI | `src/feast_features.py` | `prepare`, `demo`, `training-frame`. |

How the source data is built (`prepare`):

- `booking_id` = row index of `data/processed/hotel_bookings_clean.csv` (int64). It is only stable for a given data version (v2-clean, md5 `0bd1d207...`); a different preprocessing run would renumber the rows.
- `event_timestamp` = UTC midnight of the arrival date, built from `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`. Range: 2015-07-01 to 2017-08-31.
- `created_timestamp` = `event_timestamp`. It exists only so Feast can break ties between rows with the same key and event time; we have one row per booking, and equal values keep the output deterministic.
- The label `is_canceled` is deliberately not in the feature source. A label is the outcome, not something known about the booking when a prediction is requested. It is stored in a separate `booking_labels.parquet` and joined in through the entity dataframe at training time, which is the standard Feast pattern.
- Both parquet files and the two `.db` files are generated and gitignored; `feature_repo/data/.gitkeep` keeps the folder.

TTL: `ttl=timedelta(0)`. In a point-in-time join the ttl is how far before the requested time a feature row may be and still count; `0` means no lower bound, so the latest row at or before the requested time is used however old it is (Feast 0.66 `infra/offline_stores/dask.py`, `_filter_ttl`: with a non-zero ttl it also requires `feature_ts >= entity_ts - ttl`). Here every booking has one row and the training entity timestamps equal the feature timestamps, so any ttl would give the same training frame; `0` simply states that a booking's features never go stale within this static 2015-2017 snapshot. Two things the ttl does *not* do here, checked rather than assumed:

- The SQLite online read path in 0.66 does not apply the ttl. I re-applied the repo with `ttl=timedelta(days=3650)`, materialized, and `get_online_features` still returned booking 0 (arrival 2015-07-01, more than 3650 days before 2026-10-02) with its values. The response builder (`feast/utils.py`, `_populate_response_from_feature_data`) only sets PRESENT or NOT_FOUND; an age check (OUTSIDE_MAX_AGE) exists only in the `precompute_online` FeatureService fast path, which this project does not use.
- `feast materialize` with explicit start/end ignores it. Only `feast materialize-incremental` uses it, to pick the first window when nothing was materialized yet (`now - ttl`, or one year back when ttl is 0); either way it would miss 2015-2017 data, which is why the commands below use explicit dates.

In a live system with fresh data you would choose a ttl matching how long a value stays meaningful.

## 2. How to run

From the project root, with the venv activated (or call `.venv\Scripts\...` directly):

```powershell
python -m src.feast_features prepare                       # CSV -> feature_repo/data/*.parquet
cd feature_repo
feast apply                                                 # register entity, view, service; create the SQLite table
feast materialize 2015-07-01T00:00:00 2017-09-01T00:00:00   # offline parquet -> online SQLite (about 35 s)
cd ..
python -m src.feast_features demo                           # fetch demo + consistency check -> reports_data/feast_demo.txt
python -m src.feast_features training-frame                 # optional: point-in-time join for all 85,716 bookings
feast -c feature_repo feature-views list                    # inspect the registry
```

The script uses `FeatureStore(repo_path=<project>/feature_repo)`, so `prepare`, `demo` and `training-frame` work from any directory; only the `feast` CLI needs to run in (or `-c`) the repo directory.

## 3. Captured output

`feast apply`:

```
No project found in the repository. Using project name stayguard defined in feature_store.yaml
Applying changes for project stayguard
Created project stayguard
Created entity booking_id
Created feature view booking_features
Created feature service stayguard_model_v1

Created sqlite table stayguard_booking_features
```

`feast materialize 2015-07-01T00:00:00 2017-09-01T00:00:00` (progress bar trimmed; it completed in about 35 s):

```
Materializing 1 feature views from 2015-07-01 00:00:00+00:00 to 2017-09-01 00:00:00+00:00 into the sqlite online store.
booking_features:
```

Registry contents:

```
feast feature-views list
NAME              ENTITIES        TYPE         ENABLED    STATE
booking_features  {'booking_id'}  FeatureView  Yes        AVAILABLE_ONLINE

feast entities list
NAME        DESCRIPTION                                                              TYPE
booking_id  Synthetic booking id = row index in hotel_bookings_clean.csv (v2-clean)  ValueType.INT64

feast feature-services list
NAME                FEATURES
stayguard_model_v1  booking_features:lead_time, booking_features:adr, booking_features:previous_cancellations
```

`feast feature-views describe booking_features` shows exactly three features (`lead_time` INT64, `adr` FLOAT, `previous_cancellations` INT64) and `materializationIntervals` 2015-07-01 to 2017-09-01.

### Historical fetch (training side)

Entity dataframe: five bookings with their arrival timestamp and the `is_canceled` label.

```
 booking_id           event_timestamp  is_canceled  lead_time        adr  previous_cancellations
          0 2015-07-01 00:00:00+00:00            0        342   0.000000                       0
       1000 2015-08-12 00:00:00+00:00            1         33 191.000000                       0
      20000 2016-06-02 00:00:00+00:00            0        105  64.900002                       0
      50000 2017-07-05 00:00:00+00:00            1        169  89.400002                       0
      85715 2017-08-29 00:00:00+00:00            0        205 151.199997                       0
```

(`adr` is Float32, hence 64.900002.) The `training-frame` command joins all 85,716 bookings in under a second (0.9 s on the latest run; 85,716 rows out, 0 null feature cells), so the file offline store handles the full training set comfortably.

### Point-in-time behaviour

Booking 20000 has one feature row, stamped 2016-06-02. The demo re-runs the five-booking request with only booking 20000's timestamp moved one day earlier, to 2016-06-01:

```
 booking_id           event_timestamp  is_canceled  lead_time        adr  previous_cancellations
          0 2015-07-01 00:00:00+00:00            0        342   0.000000                       0
       1000 2015-08-12 00:00:00+00:00            1         33 191.000000                       0
      50000 2017-07-05 00:00:00+00:00            1        169  89.400002                       0
      85715 2017-08-29 00:00:00+00:00            0        205 151.199997                       0
rows requested: 5, returned: 4; rows for booking 20000: 0 (a leak would show lead_time=105 etc.); other 4 bookings unchanged: True
```

The value that only became known on 2016-06-02 is not leaked into 2016-06-01. But the entity row does not come back with null features, as Feast's left-join semantics would suggest; it disappears. A key that has no feature row at all does come back, with NaN:

```
 booking_id           event_timestamp  is_canceled  lead_time  adr  previous_cancellations
   10000000 2017-09-01 00:00:00+00:00            0        NaN  NaN                     NaN
```

Why, from the Feast 0.66.0 source (`feature_repo` uses `offline_store: type: file`, which maps to `DaskOfflineStore` in `feast/infra/offline_stores/dask.py`):

1. `_merge` (line 1108) does a `how="left"` merge of the entity dataframe with the feature rows on the join key only, not on time. Booking 20000 gets one candidate row carrying the 2016-06-02 feature timestamp; the unknown key gets a row whose feature timestamp is null.
2. `_filter_ttl` (line 1182, called at line 316) then keeps rows where the feature timestamp is null or `<= entity timestamp` (with a non-zero ttl, also `>= entity timestamp - ttl`). The unknown key survives through the null branch; booking 20000's only candidate fails the time test, and since it is the only row carrying that entity, the entity row is gone.
3. Nothing re-attaches dropped entity rows afterwards. By contrast, the created-timestamp cutoff in the same file (`_apply_created_timestamp_cutoff`, line 1218) masks too-new values instead of dropping them, with the comment "so the entity row survives", so dropping here is an inconsistency in this offline store rather than intended left-join behaviour.

I checked that it is not caused by the demo code: the result is the same with the early row alone or mixed with valid rows, with `ttl=0` and `ttl=3650 days`, with tz-aware and tz-naive entity timestamps (Feast converts naive to UTC), and with `full_feature_names=True`. By the same code path (read, not tested) an entity row whose only feature row is older than a finite ttl would also be dropped. I did not test other offline stores.

Practical consequence: a training entity dataframe can silently lose rows (here 5 in, 4 out) instead of carrying nulls, so the label distribution can shift without an error. Training code should compare row counts before and after `get_historical_features` and treat missing rows as "features not yet available" explicitly. In the full `training-frame` run every entity timestamp equals its feature timestamp, so all 85,716 rows come back.

### Online fetch (inference side)

```
store.get_online_features(features=[...], entity_rows=[{"booking_id": 20000}]).to_dict()
{'booking_id': [20000], 'lead_time': [105], 'adr': [64.9000015258789], 'previous_cancellations': [0]}
```

The same call using `features=store.get_feature_service("stayguard_model_v1")` returns identical values.

### Consistency check (booking 20000)

```
lead_time                online=       105 historical=       105 csv=     105.0 OK
adr                      online=64.9000015258789 historical=64.9000015258789 csv=      64.9 OK
previous_cancellations   online=         0 historical=         0 csv=       0.0 OK
CONSISTENCY CHECK: PASS
```

Online, historical and the raw CSV row agree. Online and historical are compared for exact equality (both read the same Float32 value, 64.9000015258789); the CSV holds float64 64.9, so it is compared with `np.isclose(rtol=1e-6, atol=0)`. Float32 keeps about 7 significant digits (relative error at most about 6e-8), so 1e-6 accepts storage rounding and flags anything larger. Across all 85,716 rows the Float32 source matches the CSV at this tolerance (largest absolute adr difference 1.5e-5). The script prints the shared definition first: both calls use the refs `booking_features:lead_time`, `booking_features:adr`, `booking_features:previous_cancellations`, resolved from the one registered `booking_features` view and the `stayguard_model_v1` service.

## 4. How Feast keeps training and inference features consistent

- **Single definition.** The feature names, types, entity, source and ttl are declared once in `features.py` and registered with `feast apply`. Training asks for `booking_features:lead_time`; the serving path asks for the same string. Nobody re-implements "lead_time" in a notebook and again in the API.
- **Point-in-time joins.** `get_historical_features` takes an entity dataframe of (key, timestamp) pairs and, per row, picks the latest feature value at or before that timestamp (and within ttl). Training data therefore reflects what was knowable then, not what is known now. The 1-day-earlier example above shows the future value being refused (in this file store the row is dropped rather than returned with nulls). This matters most for features that change over time (running counts, balances); here each booking has a single row, so the demonstration is of the mechanism rather than of a time series.
- **Avoiding training-serving skew.** Skew comes from computing features through two code paths (a SQL/pandas job for training, hand-written code in the service) that drift apart, or from leaking future data into training. With one definition and one materialization step, the online store holds the values the offline store holds (verified above for booking 20000), and point-in-time joins remove the leakage side.
- **Online and offline stores.** The offline store (Parquet here, a warehouse in production) holds full history and is optimised for bulk joins when building training sets. The online store (SQLite here, Redis/DynamoDB/Bigtable in production) holds only the latest value per key for millisecond lookup. `feast materialize` is the bridge between them, so the two are populated from the same source by the same definition.
- **Reuse.** A `FeatureService` names a set of features for a model version. A second model or team can reuse `booking_features` in its own service without copying logic, and the registry is the catalogue of what exists, with its entity and source.

## 5. Limitations

- Everything is local: file offline store, SQLite online store, local registry. SQLite and the Parquet files are not for concurrent or production serving; the same definitions would move to Redis/BigQuery-style stores by changing `feature_store.yaml`.
- `booking_id` is synthetic (row position in v2-clean). Real bookings would carry a stable reservation id from the source system, and the key would not change when the cleaning pipeline changes.
- The data is a static snapshot with one feature row per booking, so materialization is a one-off and the ttl is `0`. There is no streaming or incremental refresh in this project.
- The three features already exist as columns in the training CSV. Here Feast demonstrates the pattern; it does not replace the loader in `src/train.py`, and the trained model is not retrained from Feast output.
- The model uses 69 features, Feast serves 3. A full deployment would define views for all model inputs (including the one-hot encoded columns or, better, their raw sources with transformations declared in Feast), expose them through one `FeatureService`, and have training and the prediction service both read from it.
- Feast 0.66's file offline store drops an entity row whose only feature rows are in its future (or outside a finite ttl) instead of returning nulls (see above, `_filter_ttl` in `dask.py`); training code has to check row counts. Other offline stores were not tested.
- No Windows-specific failures occurred. Practical notes: the progress bar of `feast materialize` emits ANSI colour codes into captured logs, and relative paths in `feature_store.yaml` and `FileSource` resolve against `feature_repo/`, so the `feast` CLI must run there (or use `-c feature_repo`).
