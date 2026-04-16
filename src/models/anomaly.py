from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.utils.math import minmax


def isolation_forest_score(df: pd.DataFrame, contamination: float = 0.08, seed: int = 42) -> pd.Series:
    if len(df) < 8:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(df)
    score = -model.decision_function(df)
    return minmax(pd.Series(score, index=df.index, dtype=float))


def lof_score(df: pd.DataFrame, n_neighbors: int | None = None) -> pd.Series:
    if len(df) < 8:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    k = n_neighbors or min(25, max(5, len(df) // 10))
    model = LocalOutlierFactor(n_neighbors=k)
    model.fit(df)
    score = -model.negative_outlier_factor_
    return minmax(pd.Series(score, index=df.index, dtype=float))
