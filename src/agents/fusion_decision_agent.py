from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.config import Settings
from src.models.fusion import weighted_fusion


class FusionDecisionAgent(BaseAgent):
    name = "fusion_decision"

    def __init__(self, settings: Settings):
        self.settings = settings

    def merge_scores(self, features_df: pd.DataFrame, agent_outputs: list[pd.DataFrame]) -> pd.DataFrame:
        merged = features_df.copy()
        for out in agent_outputs:
            merged = merged.merge(out, on="transaction_id", how="left")
        return merged

    def compute_final_score(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        df = merged_df.copy()
        score_cols = [
            "transaction_behavior_score",
            "temporal_sequence_score",
            "geospatial_score",
            "communication_risk_score",
            "novelty_drift_score",
        ]
        for col in score_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0).astype(float).clip(0, 1)

        df["final_risk_score"] = weighted_fusion(df, self.settings.agent_weights).clip(0, 1)

        reason_pairs = [
            ("transaction_behavior_score", "transaction_behavior_reason"),
            ("temporal_sequence_score", "temporal_sequence_reason"),
            ("geospatial_score", "geospatial_reason"),
            ("communication_risk_score", "communication_risk_reason"),
            ("novelty_drift_score", "novelty_drift_reason"),
        ]
        top_reasons: list[str] = []
        for _, row in df.iterrows():
            candidates: list[tuple[float, str]] = []
            for score_col, reason_col in reason_pairs:
                reason = str(row.get(reason_col, "")).strip()
                score = float(row.get(score_col, 0.0))
                if reason:
                    candidates.append((score, reason))
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_reasons.append(" | ".join(reason for _, reason in candidates[:3]))
        df["top_risk_reasons"] = top_reasons
        return df

    def choose_threshold(self, scored_df: pd.DataFrame) -> float:
        scores = scored_df["final_risk_score"].fillna(0.0).astype(float)
        if scores.empty:
            return 1.0

        q = 1.0 - float(self.settings.target_flag_rate)
        quantile_threshold = float(scores.quantile(q))
        threshold = max(float(self.settings.score_floor), quantile_threshold)
        threshold = max(float(self.settings.min_score), min(threshold, float(self.settings.max_score)))
        return threshold

    def _apply_bounds(self, ranked_df: pd.DataFrame, flagged_series: pd.Series) -> pd.Series:
        n = len(ranked_df)
        if n <= 1:
            return pd.Series([True] * n, index=ranked_df.index)

        min_flags = max(1, int(math.ceil(n * self.settings.min_flag_rate)))
        max_flags = min(n - 1, int(math.floor(n * self.settings.max_flag_rate)))
        if max_flags < min_flags:
            max_flags = min_flags

        count = int(flagged_series.sum())
        adjusted = flagged_series.copy()

        if count < min_flags:
            adjusted.iloc[:min_flags] = True
            count = int(adjusted.sum())

        if count > max_flags:
            keep_ids = set(ranked_df.iloc[:max_flags]["transaction_id"].astype(str).tolist())
            adjusted = ranked_df["transaction_id"].astype(str).isin(keep_ids)

        return adjusted

    def flag_transactions(self, scored_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        df = scored_df.copy().sort_values(["final_risk_score", "transaction_id"], ascending=[False, True]).reset_index(drop=True)
        raw_flagged = df["final_risk_score"].astype(float) >= float(threshold)
        bounded = self._apply_bounds(df, raw_flagged)

        df["flagged"] = bounded.astype(bool)
        df["threshold_used"] = float(threshold)
        return df

    def run(self, features_df: pd.DataFrame, agent_outputs: list[pd.DataFrame]) -> pd.DataFrame:
        merged = self.merge_scores(features_df, agent_outputs)
        scored = self.compute_final_score(merged)
        threshold = self.choose_threshold(scored)
        flagged = self.flag_transactions(scored, threshold)
        return flagged[["transaction_id", "final_risk_score", "flagged", "top_risk_reasons"]]
