"""Load the raw hotel_bookings.csv from data/raw.

`EXPECTED_COLUMNS` is the schema contract for the raw Kaggle file; it maps each
column to the dtype *kind* (numpy kind code: 'i' int, 'f' float, 'O' object)
that pandas infers for it. `validate.py` checks column presence and dtype kind.
"""
from pathlib import Path

import pandas as pd

from src.utils import DATA_RAW

RAW_FILE = DATA_RAW / "hotel_bookings.csv"

# column -> expected dtype kind. 'f' columns hold nulls in the raw file
# (children, agent, company), so pandas reads them as float.
EXPECTED_COLUMNS: dict[str, str] = {
    "hotel": "O",
    "is_canceled": "i",
    "lead_time": "i",
    "arrival_date_year": "i",
    "arrival_date_month": "O",
    "arrival_date_week_number": "i",
    "arrival_date_day_of_month": "i",
    "stays_in_weekend_nights": "i",
    "stays_in_week_nights": "i",
    "adults": "i",
    "children": "f",
    "babies": "i",
    "meal": "O",
    "country": "O",
    "market_segment": "O",
    "distribution_channel": "O",
    "is_repeated_guest": "i",
    "previous_cancellations": "i",
    "previous_bookings_not_canceled": "i",
    "reserved_room_type": "O",
    "assigned_room_type": "O",
    "booking_changes": "i",
    "deposit_type": "O",
    "agent": "f",
    "company": "f",
    "days_in_waiting_list": "i",
    "customer_type": "O",
    "adr": "f",
    "required_car_parking_spaces": "i",
    "total_of_special_requests": "i",
    "reservation_status": "O",
    "reservation_status_date": "O",
}


def load_raw(path: Path | str = RAW_FILE) -> pd.DataFrame:
    """Read the raw bookings CSV into a DataFrame (no cleaning)."""
    return pd.read_csv(path)
