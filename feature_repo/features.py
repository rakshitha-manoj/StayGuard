"""Feast definitions for StayGuard: one entity, one file source, one view, one service.

Applied with `feast apply` from feature_repo/. Both training (get_historical_features)
and inference (get_online_features) read these same objects.
"""
from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

# Synthetic key: row position in the v2-clean CSV (see src/feast_features.py).
booking = Entity(
    name="booking_id",
    join_keys=["booking_id"],
    value_type=ValueType.INT64,
    description="Synthetic booking id = row index in hotel_bookings_clean.csv (v2-clean)",
)

booking_source = FileSource(
    name="booking_features_source",
    path="data/booking_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

booking_features = FeatureView(
    name="booking_features",
    entities=[booking],
    # ttl bounds how far before the entity timestamp a point-in-time join may look
    # for a feature row. timedelta(0) means no lower bound: the latest row at or
    # before the entity timestamp is used however old it is. That fits a static
    # 2015-2017 snapshot with one row per booking. (Feast 0.66's SQLite online read
    # does not apply ttl; materialize-incremental would use it for its first window.)
    ttl=timedelta(0),
    schema=[
        Field(name="lead_time", dtype=Int64),
        Field(name="adr", dtype=Float32),
        Field(name="previous_cancellations", dtype=Int64),
    ],
    source=booking_source,
    online=True,
)

# Named bundle of features a model consumes; training and serving both reference it.
stayguard_model_v1 = FeatureService(
    name="stayguard_model_v1",
    features=[booking_features],
)
