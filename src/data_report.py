"""Compare dataset v1 (raw) and v2 (clean) from the actual files; print a markdown table.

CLI: python -m src.data_report [--out report/dvc_stats.md]
DVC md5s are read from data/raw/hotel_bookings.csv.dvc (v1) and dvc.lock (v2).
"""
import argparse

import pandas as pd
import yaml

from src.ingest import RAW_FILE, load_raw
from src.preprocess import OUT_FILE
from src.utils import PROJECT_ROOT

AFFECTED_NULL_COLUMNS = ["children", "country", "agent", "company"]


def dvc_md5_raw() -> str:
    """md5 of the raw CSV from its .dvc pointer file."""
    meta = yaml.safe_load((PROJECT_ROOT / "data/raw/hotel_bookings.csv.dvc").read_text())
    return meta["outs"][0]["md5"]


def dvc_md5_clean() -> str:
    """md5 of the processed CSV from dvc.lock (preprocess stage output)."""
    lock = yaml.safe_load((PROJECT_ROOT / "dvc.lock").read_text())
    outs = lock["stages"]["preprocess"]["outs"]
    return next(o["md5"] for o in outs if o["path"].endswith("hotel_bookings_clean.csv"))


def stats(df: pd.DataFrame, path, md5: str) -> dict:
    """Summary statistics for one dataset version (handles raw and clean schemas)."""
    guests = df["total_guests"] if "total_guests" in df else (
        df["adults"] + df["children"].fillna(0) + df["babies"])
    size = path.stat().st_size
    s = {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Total nulls": int(df.isna().sum().sum()),
    }
    for c in AFFECTED_NULL_COLUMNS:
        s[f"Nulls: {c}"] = int(df[c].isna().sum()) if c in df else "n/a (column removed or encoded)"
    s.update({
        "File size (bytes)": size,
        "File size (MB)": round(size / 1e6, 2),
        "DVC md5": md5,
        "Cancel rate": round(float(df["is_canceled"].mean()), 4),
        "adr mean": round(float(df["adr"].mean()), 2),
        "adr median": round(float(df["adr"].median()), 2),
        "adr min": round(float(df["adr"].min()), 2),
        "adr max": round(float(df["adr"].max()), 2),
        "lead_time mean": round(float(df["lead_time"].mean()), 2),
        "lead_time max": int(df["lead_time"].max()),
        "Zero-guest rows": int((guests == 0).sum()),
        "Negative adr rows": int((df["adr"] < 0).sum()),
        "Duplicate rows": int(df.duplicated().sum()),
    })
    return s


def markdown_table(v1: dict, v2: dict) -> str:
    lines = ["| Metric | v1-raw | v2-clean |", "|---|---|---|"]
    for k in v1:
        lines.append(f"| {k} | {v1[k]:,} | {v2[k]:,} |" if isinstance(v1[k], int) and isinstance(v2[k], int)
                     else f"| {k} | {v1[k]} | {v2[k]} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="also write the table to this file")
    args = parser.parse_args()
    v1 = stats(load_raw(RAW_FILE), RAW_FILE, dvc_md5_raw())
    v2 = stats(pd.read_csv(OUT_FILE), OUT_FILE, dvc_md5_clean())
    table = markdown_table(v1, v2)
    print(table)
    if args.out:
        (PROJECT_ROOT / args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
