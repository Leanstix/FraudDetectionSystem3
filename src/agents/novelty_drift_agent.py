from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.models.anomaly import isolation_forest_score, lof_score


class NoveltyDriftAgent(BaseAgent):
    name = "novelty_drift"

    def run(self, features_df: pd.DataFrame) -> pd.DataFrame:
        df = features_df.copy()
        numeric_cols = [
            c
            for c in [
                "amount",
                "amount_robust_z_sender",
                "amount_robust_z_recipient",
                "txn_count_past_1h",
                "txn_count_past_24h",
                "txn_count_past_7d",
                "distance_from_residence_km",
                "distance_from_latest_gps_km",
                "suspicious_communication_window_score",
                "novelty_score",
                "reference_geo_novelty",
                "reference_hour_rarity",
                "reference_weekday_rarity",
            ]
            if c in df.columns
        ]

        if numeric_cols:
            x = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
            iso = isolation_forest_score(x, contamination=0.08, seed=42)
            lof = lof_score(x)
            base = df.get("novelty_score", pd.Series(np.zeros(len(df)))).astype(float)
            unseen_mix = (
                df.get("unseen_transaction_type_indicator", 0).astype(float)
                + df.get("unseen_payment_method_indicator", 0).astype(float)
                + df.get("unseen_location_pattern_indicator", 0).astype(float)
            ) / 3.0
            score = (0.30 * iso + 0.30 * lof + 0.25 * base + 0.15 * unseen_mix).clip(0, 1)
        else:
            score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

        reasons = []
        for i in range(len(df)):
            parts = []
            if score.iloc[i] > 0.8:
                parts.append("high_unsupervised_outlier")
            if float(df.get("novelty_score", pd.Series(np.zeros(len(df)))).iloc[i]) > 0.7:
                parts.append("feature_novelty")
            reasons.append(";".join(parts) if parts else "stable_profile")

        return pd.DataFrame(
            {
                "transaction_id": df["transaction_id"],
                "novelty_drift_score": score,
                "novelty_drift_reason": reasons,
            }
        )
