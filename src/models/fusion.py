from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_fusion(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if not weights:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    total = 0.0
    acc = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for col, w in weights.items():
        if col not in df.columns:
            continue
        total += float(w)
        acc = acc + df[col].astype(float).fillna(0.0) * float(w)
    if total <= 0:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return acc / total
