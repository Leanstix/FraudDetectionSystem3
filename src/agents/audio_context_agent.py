from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.llm.audio_reasoner import AudioReasoner


class AudioContextAgent(BaseAgent):
    name = "audio_context"

    def __init__(self, audio_reasoner: AudioReasoner):
        self.audio_reasoner = audio_reasoner

    def score_audio_events(self, audio_df: pd.DataFrame, dataset_name: str) -> dict:
        scored: dict[str, dict] = {}
        for _, row in audio_df.iterrows():
            payload = self.audio_reasoner.analyze_audio_event(row.to_dict(), dataset_name=dataset_name)
            scored[str(row.get("audio_id"))] = payload
        return scored

    def run(self, features_df: pd.DataFrame, audio_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        out = features_df[["transaction_id", "timestamp", "sender_full_name"]].copy()
        out["audio_context_score"] = 0.0
        out["audio_context_reason"] = "no_audio_context"

        if audio_df.empty:
            return out[["transaction_id", "audio_context_score", "audio_context_reason"]]

        event_scores = self.score_audio_events(audio_df, dataset_name)
        events = audio_df.copy()
        events["event_score"] = events["audio_id"].astype(str).map(
            lambda k: float(event_scores.get(k, {}).get("risk_score", 0.0))
        )
        events["event_reason"] = events["audio_id"].astype(str).map(
            lambda k: str(event_scores.get(k, {}).get("reason", "audio_event"))
        )

        for idx, row in out.iterrows():
            tx_ts = row.get("timestamp")
            if pd.isna(tx_ts):
                continue

            sender = str(row.get("sender_full_name") or "").strip().lower()
            if sender:
                candidate = events[events["speaker_norm"].str.contains(re.escape(sender), regex=True, na=False)]
            else:
                candidate = events

            candidate = candidate[candidate["timestamp"] <= tx_ts]
            if candidate.empty:
                continue

            hours = (tx_ts - candidate["timestamp"]).dt.total_seconds() / 3600.0
            decay = np.exp(-hours / 72.0)
            combined = candidate["event_score"].fillna(0.0).values * decay.values
            best_idx = int(np.argmax(combined))
            best_score = float(np.clip(combined[best_idx], 0.0, 1.0))
            best_reason = str(candidate.iloc[best_idx]["event_reason"])

            out.at[idx, "audio_context_score"] = best_score
            out.at[idx, "audio_context_reason"] = f"audio:{best_reason[:140]}"

        return out[["transaction_id", "audio_context_score", "audio_context_reason"]]
