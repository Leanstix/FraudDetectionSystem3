from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.constants import SUSPICIOUS_KEYWORDS
from src.types import BaselineArtifacts


def _robust_z(value: float, median: float, mad: float) -> float:
    if pd.isna(value) or pd.isna(median):
        return 0.0
    if pd.isna(mad) or mad == 0:
        return 0.0
    return float(0.6745 * (value - median) / mad)


class BaselineStore:
    def __init__(self):
        self.reference_transactions_df = pd.DataFrame()
        self.reference_users_df = pd.DataFrame()
        self.reference_locations_df = pd.DataFrame()
        self.reference_sms_df = pd.DataFrame()
        self.reference_mails_df = pd.DataFrame()

        self.sender_profiles_df = pd.DataFrame()
        self.recipient_profiles_df = pd.DataFrame()
        self.pair_profiles_df = pd.DataFrame()
        self.geo_profiles_df = pd.DataFrame()
        self.temporal_profiles_df = pd.DataFrame()
        self.communication_profiles_df = pd.DataFrame()

    def fit(
        self,
        reference_transactions_df: pd.DataFrame,
        reference_users_df: pd.DataFrame,
        reference_locations_df: pd.DataFrame,
        reference_sms_df: pd.DataFrame,
        reference_mails_df: pd.DataFrame,
    ) -> "BaselineStore":
        self.reference_transactions_df = reference_transactions_df.copy()
        self.reference_users_df = reference_users_df.copy()
        self.reference_locations_df = reference_locations_df.copy()
        self.reference_sms_df = reference_sms_df.copy()
        self.reference_mails_df = reference_mails_df.copy()

        self.sender_profiles_df = self.build_sender_profiles()
        self.recipient_profiles_df = self.build_recipient_profiles()
        self.pair_profiles_df = self.build_pair_profiles()
        self.geo_profiles_df = self.build_geo_profiles()
        self.temporal_profiles_df = self.build_temporal_profiles()
        self.communication_profiles_df = self.build_communication_profiles()
        return self

    def build_sender_profiles(self) -> pd.DataFrame:
        tx = self.reference_transactions_df.copy()
        tx["hour"] = pd.to_datetime(tx["timestamp"], errors="coerce", utc=True).dt.hour
        grouped = tx.groupby("sender_id", dropna=False)

        rows = []
        for sender_id, grp in grouped:
            methods = sorted({str(v) for v in grp["payment_method"].dropna().astype(str) if v})
            types = sorted({str(v) for v in grp["transaction_type"].dropna().astype(str) if v})
            hours = sorted({int(v) for v in grp["hour"].dropna().astype(int).tolist()})
            rows.append(
                {
                    "sender_id": sender_id,
                    "ref_sender_tx_count": int(len(grp)),
                    "ref_sender_amount_median": float(grp["amount"].median()),
                    "ref_sender_amount_mad": float((grp["amount"] - grp["amount"].median()).abs().median()),
                    "ref_sender_payment_methods": "|".join(methods),
                    "ref_sender_transaction_types": "|".join(types),
                    "ref_sender_hours": "|".join(str(v) for v in hours),
                }
            )
        return pd.DataFrame(rows)

    def build_recipient_profiles(self) -> pd.DataFrame:
        tx = self.reference_transactions_df
        grouped = tx.groupby("recipient_id", dropna=False)
        rows = []
        for recipient_id, grp in grouped:
            rows.append(
                {
                    "recipient_id": recipient_id,
                    "ref_recipient_tx_count": int(len(grp)),
                    "ref_recipient_amount_median": float(grp["amount"].median()),
                    "ref_recipient_amount_mad": float((grp["amount"] - grp["amount"].median()).abs().median()),
                }
            )
        return pd.DataFrame(rows)

    def build_pair_profiles(self) -> pd.DataFrame:
        tx = self.reference_transactions_df
        grouped = tx.groupby(["sender_id", "recipient_id"], dropna=False)
        rows = []
        for (sender_id, recipient_id), grp in grouped:
            rows.append(
                {
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "ref_pair_count": int(len(grp)),
                    "ref_pair_amount_median": float(grp["amount"].median()),
                }
            )
        return pd.DataFrame(rows)

    def build_geo_profiles(self) -> pd.DataFrame:
        tx = self.reference_transactions_df
        city_series = tx["location"].fillna("<missing>").astype(str)
        freq = city_series.value_counts(normalize=True)
        rows = []
        for city, prob in freq.items():
            rows.append({"location": city, "ref_geo_prob": float(prob)})
        return pd.DataFrame(rows)

    def build_temporal_profiles(self) -> pd.DataFrame:
        tx = self.reference_transactions_df.copy()
        tx["hour"] = pd.to_datetime(tx["timestamp"], errors="coerce", utc=True).dt.hour
        tx["weekday"] = pd.to_datetime(tx["timestamp"], errors="coerce", utc=True).dt.weekday

        hour_prob = tx["hour"].value_counts(normalize=True).to_dict()
        weekday_prob = tx["weekday"].value_counts(normalize=True).to_dict()
        rows = []
        for hour in range(24):
            rows.append(
                {
                    "hour": hour,
                    "ref_hour_prob": float(hour_prob.get(hour, 0.0)),
                    "ref_weekday_prob_mean": float(np.mean(list(weekday_prob.values()) or [0.0])),
                }
            )
        temporal_df = pd.DataFrame(rows)
        temporal_df.attrs["weekday_prob"] = weekday_prob
        return temporal_df

    def build_communication_profiles(self) -> pd.DataFrame:
        messages: list[str] = []
        if not self.reference_sms_df.empty and "message_text" in self.reference_sms_df.columns:
            messages.extend(self.reference_sms_df["message_text"].fillna("").astype(str).tolist())
        if not self.reference_mails_df.empty:
            body = self.reference_mails_df.get("body_text", pd.Series(dtype=str)).fillna("").astype(str)
            subject = self.reference_mails_df.get("subject", pd.Series(dtype=str)).fillna("").astype(str)
            messages.extend((subject + " " + body).tolist())

        all_text = "\n".join(messages).lower()
        total = max(len(messages), 1)
        rows = []
        for kw in SUSPICIOUS_KEYWORDS:
            count = len(re.findall(re.escape(kw), all_text))
            rows.append({"keyword": kw, "reference_rate": float(count / total)})
        return pd.DataFrame(rows)

    def transform_target(self, target_transactions_df: pd.DataFrame) -> pd.DataFrame:
        df = target_transactions_df.copy()
        df["location"] = df["location"].fillna("<missing>").astype(str)
        df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.hour
        df["weekday"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.weekday

        if not self.sender_profiles_df.empty:
            df = df.merge(self.sender_profiles_df, on="sender_id", how="left")
        if not self.recipient_profiles_df.empty:
            df = df.merge(self.recipient_profiles_df, on="recipient_id", how="left")
        if not self.pair_profiles_df.empty:
            df = df.merge(self.pair_profiles_df, on=["sender_id", "recipient_id"], how="left")
        if not self.geo_profiles_df.empty:
            df = df.merge(self.geo_profiles_df, on="location", how="left")

        hour_lookup = {}
        weekday_lookup = {}
        if not self.temporal_profiles_df.empty:
            hour_lookup = {
                int(row["hour"]): float(row["ref_hour_prob"])
                for _, row in self.temporal_profiles_df.iterrows()
            }
            weekday_lookup = self.temporal_profiles_df.attrs.get("weekday_prob", {})

        df["ref_sender_amount_robust_z"] = df.apply(
            lambda r: _robust_z(
                float(r.get("amount", 0.0)),
                float(r.get("ref_sender_amount_median", np.nan)),
                float(r.get("ref_sender_amount_mad", np.nan)),
            ),
            axis=1,
        )
        df["ref_recipient_amount_robust_z"] = df.apply(
            lambda r: _robust_z(
                float(r.get("amount", 0.0)),
                float(r.get("ref_recipient_amount_median", np.nan)),
                float(r.get("ref_recipient_amount_mad", np.nan)),
            ),
            axis=1,
        )

        df["pair_seen_in_reference"] = df["ref_pair_count"].fillna(0).astype(float).gt(0).astype(int)

        def _in_set(value: str, packed: str) -> int:
            if pd.isna(value) or pd.isna(packed):
                return 0
            return int(str(value) in {t for t in str(packed).split("|") if t})

        df["payment_method_seen_by_sender_ref"] = df.apply(
            lambda r: _in_set(r.get("payment_method"), r.get("ref_sender_payment_methods")),
            axis=1,
        )
        df["transaction_type_seen_by_sender_ref"] = df.apply(
            lambda r: _in_set(r.get("transaction_type"), r.get("ref_sender_transaction_types")),
            axis=1,
        )

        df["reference_hour_rarity"] = 1.0 - df["hour"].map(lambda v: hour_lookup.get(int(v), 0.0) if not pd.isna(v) else 0.0)
        df["reference_weekday_rarity"] = 1.0 - df["weekday"].map(
            lambda v: float(weekday_lookup.get(int(v), 0.0)) if not pd.isna(v) else 0.0
        )
        df["reference_geo_novelty"] = 1.0 - df["ref_geo_prob"].fillna(0.0)

        df["unseen_transaction_type_indicator"] = (1 - df["transaction_type_seen_by_sender_ref"]).clip(0, 1)
        df["unseen_payment_method_indicator"] = (1 - df["payment_method_seen_by_sender_ref"]).clip(0, 1)
        df["unseen_location_pattern_indicator"] = df["ref_geo_prob"].fillna(0.0).eq(0).astype(int)

        result_cols = [
            "transaction_id",
            "ref_sender_amount_robust_z",
            "ref_recipient_amount_robust_z",
            "pair_seen_in_reference",
            "payment_method_seen_by_sender_ref",
            "transaction_type_seen_by_sender_ref",
            "reference_hour_rarity",
            "reference_weekday_rarity",
            "reference_geo_novelty",
            "unseen_transaction_type_indicator",
            "unseen_payment_method_indicator",
            "unseen_location_pattern_indicator",
        ]
        return df[result_cols].copy()

    def export_artifacts(self) -> BaselineArtifacts:
        return BaselineArtifacts(
            sender_profiles=self.sender_profiles_df.copy(),
            recipient_profiles=self.recipient_profiles_df.copy(),
            pair_profiles=self.pair_profiles_df.copy(),
            geo_profiles=self.geo_profiles_df.copy(),
            temporal_profiles=self.temporal_profiles_df.copy(),
            communication_profiles=self.communication_profiles_df.copy(),
            metadata={
                "reference_transaction_count": int(len(self.reference_transactions_df)),
            },
        )
