from __future__ import annotations

import pandas as pd


def parse_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def hours_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return float("inf")
    delta = later - earlier
    return max(delta.total_seconds() / 3600.0, 0.0)
