from __future__ import annotations

import math

import numpy as np
import pandas as pd


def choose_target_count(n: int, target_rate: float, min_rate: float, max_rate: float) -> int:
    if n <= 1:
        return 1
    min_count = max(1, int(math.ceil(n * min_rate)))
    max_count = min(n - 1, int(math.floor(n * max_rate)))
    target = int(round(n * target_rate))
    return int(np.clip(target, min_count, max_count))


def threshold_from_target(scores: pd.Series, target_count: int) -> float:
    if scores.empty:
        return 1.0
    sorted_scores = scores.sort_values(ascending=False).reset_index(drop=True)
    idx = int(np.clip(target_count - 1, 0, len(sorted_scores) - 1))
    return float(sorted_scores.iloc[idx])
