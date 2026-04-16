from __future__ import annotations

import numpy as np
import pandas as pd


def robust_zscore(values: pd.Series) -> pd.Series:
    med = values.median()
    mad = (values - med).abs().median()
    if mad == 0 or np.isnan(mad):
        std = values.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
        return (values - values.mean()) / std
    return 0.6745 * (values - med) / mad


def minmax(series: pd.Series) -> pd.Series:
    lo = series.min()
    hi = series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - lo) / (hi - lo)
