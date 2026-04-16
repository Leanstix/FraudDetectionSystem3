from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.utils.math import minmax


class TemporalSequenceAgent(BaseAgent):
    name = "temporal_sequence"

    def run(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()

        c1h = minmax(df.get("txn_count_past_1h", pd.Series(np.zeros(len(df)))) .astype(float))
        c24h = minmax(df.get("txn_count_past_24h", pd.Series(np.zeros(len(df)))) .astype(float))
        c7d = minmax(df.get("txn_count_past_7d", pd.Series(np.zeros(len(df)))) .astype(float))
        burst = minmax(df.get("burst_count_10min", pd.Series(np.zeros(len(df)))) .astype(float))

        time_since = df.get("time_since_prev_txn_seconds", pd.Series(np.full(len(df), 9e9))).astype(float)
        recentness = 1.0 - minmax(time_since.clip(lower=0.0))

        hour_rarity = df.get("hour_rarity", pd.Series(np.zeros(len(df)))).astype(float).clip(0, 1)
        weekday_rarity = df.get("weekday_rarity", pd.Series(np.zeros(len(df)))).astype(float).clip(0, 1)
        ref_hour_rarity = df.get("reference_hour_rarity", pd.Series(np.zeros(len(df)))).astype(float).clip(0, 1)
        ref_weekday_rarity = df.get("reference_weekday_rarity", pd.Series(np.zeros(len(df)))).astype(float).clip(0, 1)

        score = (
            0.18 * c1h
            + 0.14 * c24h
            + 0.08 * c7d
            + 0.18 * recentness
            + 0.12 * burst
            + 0.10 * hour_rarity
            + 0.07 * weekday_rarity
            + 0.08 * ref_hour_rarity
            + 0.05 * ref_weekday_rarity
        ).clip(0, 1)

        reasons = []
        for i in range(len(df)):
            parts = []
            if c1h.iloc[i] > 0.7:
                parts.append("high_1h_frequency")
            if burst.iloc[i] > 0.7:
                parts.append("burst_cluster")
            if recentness.iloc[i] > 0.8:
                parts.append("rapid_sequence")
            if hour_rarity.iloc[i] > 0.8:
                parts.append("unusual_hour")
            if weekday_rarity.iloc[i] > 0.8:
                parts.append("unusual_weekday")
            if ref_hour_rarity.iloc[i] > 0.8:
                parts.append("rare_vs_reference_hour")
            if ref_weekday_rarity.iloc[i] > 0.8:
                parts.append("rare_vs_reference_weekday")
            reasons.append(";".join(parts[:4]) if parts else "typical_temporal_pattern")

        return pd.DataFrame(
            {
                "transaction_id": df["transaction_id"],
                "temporal_sequence_score": score,
                "temporal_sequence_reason": reasons,
            }
        )
