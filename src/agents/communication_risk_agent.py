from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.llm.communication_analyzer import CommunicationAnalyzer


class CommunicationRiskAgent(BaseAgent):
    name = "communication_risk"

    def __init__(self, analyzer: CommunicationAnalyzer):
        self.analyzer = analyzer

    def _half_life_hours(self) -> float:
        settings = getattr(getattr(self.analyzer, "llm_client", None), "settings", None)
        value = getattr(settings, "suspicious_comm_half_life_hours", 72.0)
        try:
            return max(1.0, float(value))
        except Exception:
            return 72.0

    def score_threads(self, sms_df: pd.DataFrame, mails_df: pd.DataFrame, dataset_name: str) -> dict:
        scored_sms: dict[str, dict] = {}
        scored_mails: dict[str, dict] = {}

        for thread_id, grp in sms_df.groupby("thread_id"):
            text_parts = [str(v) for v in grp["message_text"].astype(str).tolist()]
            text = "\n".join(text_parts)
            scored_sms[str(thread_id)] = self.analyzer.analyze_sms_thread(text, dataset_name=dataset_name)

        for thread_id, grp in mails_df.groupby("thread_id"):
            text = (grp["subject"].fillna("") + "\n" + grp["body_text"].fillna("")).str.cat(sep="\n")
            scored_mails[str(thread_id)] = self.analyzer.analyze_mail_thread(str(text), dataset_name=dataset_name)

        return {"sms": scored_sms, "mails": scored_mails}

    def run(self, features_df: pd.DataFrame, sms_df: pd.DataFrame, mails_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        thread_scores = self.score_threads(sms_df, mails_df, dataset_name)
        half_life = self._half_life_hours()

        events: list[dict] = []
        for thread_id, grp in sms_df.groupby("thread_id"):
            score_payload = thread_scores["sms"].get(str(thread_id), {})
            ts = grp["timestamp"].dropna().min()
            participants = [str(v) for v in grp[["sender", "recipient"]].fillna("").astype(str).values.ravel().tolist()]
            participant_text = " ".join(participants).lower()
            events.append(
                {
                    "timestamp": ts,
                    "participants": participant_text,
                    "risk_score": float(score_payload.get("risk_score", 0.0)),
                    "explanation": str(score_payload.get("explanation", "")),
                    "source": "sms",
                }
            )

        for thread_id, grp in mails_df.groupby("thread_id"):
            score_payload = thread_scores["mails"].get(str(thread_id), {})
            ts = grp["timestamp"].dropna().min()
            participants = [str(v) for v in grp[["sender", "recipient", "subject"]].fillna("").astype(str).values.ravel().tolist()]
            participant_text = " ".join(participants).lower()
            events.append(
                {
                    "timestamp": ts,
                    "participants": participant_text,
                    "risk_score": float(score_payload.get("risk_score", 0.0)),
                    "explanation": str(score_payload.get("explanation", "")),
                    "source": "mail",
                }
            )

        events_df = pd.DataFrame(events)
        if not events_df.empty:
            events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce")
            events_df = events_df.dropna(subset=["timestamp"]).copy()

        out = features_df[["transaction_id", "timestamp", "sender_first_name", "recipient_first_name"]].copy()
        out["communication_risk_score"] = 0.0
        out["communication_risk_reason"] = "no_risky_communication_context"

        if events_df.empty:
            return out[["transaction_id", "communication_risk_score", "communication_risk_reason"]]

        for idx, row in out.iterrows():
            ts = row.get("timestamp")
            if pd.isna(ts):
                continue

            names = [
                str(row.get("sender_first_name") or "").strip().lower(),
                str(row.get("recipient_first_name") or "").strip().lower(),
            ]
            names = [n for n in names if n]
            if not names:
                continue

            mask = pd.Series(False, index=events_df.index)
            for name in names:
                mask = mask | events_df["participants"].str.contains(re.escape(name), regex=True)

            scoped = events_df[mask & (events_df["timestamp"] <= ts) & (events_df["timestamp"] >= ts - pd.Timedelta(days=21))]
            if scoped.empty:
                continue

            hours = (ts - scoped["timestamp"]).dt.total_seconds() / 3600.0
            decayed = scoped["risk_score"].values * np.exp(-hours.values / half_life)
            best_idx = int(np.argmax(decayed))
            best_score = float(np.clip(decayed[best_idx], 0.0, 1.0))
            best_row = scoped.iloc[best_idx]
            out.at[idx, "communication_risk_score"] = best_score
            out.at[idx, "communication_risk_reason"] = f"{best_row['source']}:{best_row['explanation'][:120]}"

        return out[["transaction_id", "communication_risk_score", "communication_risk_reason"]]
