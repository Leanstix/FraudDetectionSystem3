from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.constants import SUSPICIOUS_KEYWORDS
from src.data.baseline_store import BaselineStore
from src.utils.geo import haversine_km
from src.utils.math import minmax, robust_zscore


class FeatureStore:
    def __init__(self, baseline_store: BaselineStore | None = None):
        self.baseline_store = baseline_store

    def build_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        df = transactions_df.copy().sort_values("timestamp").reset_index(drop=True)

        df["amount_robust_z_sender"] = df.groupby("sender_id", dropna=False)["amount"].transform(
            lambda s: robust_zscore(s.astype(float).fillna(0.0))
        )
        df["amount_robust_z_recipient"] = df.groupby("recipient_id", dropna=False)["amount"].transform(
            lambda s: robust_zscore(s.astype(float).fillna(0.0))
        )

        pair_key = df["sender_id"].fillna("") + "->" + df["recipient_id"].fillna("")
        df["new_sender_recipient_pair"] = (pair_key.groupby(pair_key).cumcount() == 0).astype(int)

        method_key = df["sender_id"].fillna("") + "::" + df["payment_method"].fillna("")
        tx_type_key = df["sender_id"].fillna("") + "::" + df["transaction_type"].fillna("")
        df["new_payment_method_for_sender"] = (method_key.groupby(method_key).cumcount() == 0).astype(int)
        df["new_transaction_type_for_sender"] = (tx_type_key.groupby(tx_type_key).cumcount() == 0).astype(int)

        desc = df["description"].fillna("").astype(str)
        df["description_present"] = desc.str.strip().ne("").astype(int)
        df["description_length"] = desc.str.len()
        kw_pattern = "|".join(re.escape(k) for k in SUSPICIOUS_KEYWORDS)
        df["description_keyword_hits"] = desc.str.lower().str.count(kw_pattern)

        return df

    def build_temporal_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        df = transactions_df[["transaction_id", "sender_id", "timestamp"]].copy()
        out = df.copy().sort_values(["sender_id", "timestamp"]).reset_index(drop=True)

        out["txn_count_past_1h"] = 0.0
        out["txn_count_past_24h"] = 0.0
        out["txn_count_past_7d"] = 0.0
        out["time_since_prev_txn_seconds"] = np.nan
        out["hour_rarity"] = 0.0
        out["weekday_rarity"] = 0.0
        out["burst_count_10min"] = 0.0

        for _, grp in out.groupby("sender_id", dropna=False):
            idx = grp.index
            g = grp.set_index("timestamp").sort_index()

            c1h = g["transaction_id"].rolling("1h").count() - 1
            c24h = g["transaction_id"].rolling("24h").count() - 1
            c7d = g["transaction_id"].rolling("7D").count() - 1
            burst = g["transaction_id"].rolling("10min").count() - 1

            out.loc[idx, "txn_count_past_1h"] = c1h.values
            out.loc[idx, "txn_count_past_24h"] = c24h.values
            out.loc[idx, "txn_count_past_7d"] = c7d.values
            out.loc[idx, "burst_count_10min"] = burst.values

            ts = grp["timestamp"].sort_values()
            delta = ts.diff().dt.total_seconds().fillna(np.inf)
            out.loc[ts.index, "time_since_prev_txn_seconds"] = delta.values

            hours = grp["timestamp"].dt.hour
            weekdays = grp["timestamp"].dt.weekday
            hour_freq = hours.value_counts(normalize=True)
            weekday_freq = weekdays.value_counts(normalize=True)
            out.loc[idx, "hour_rarity"] = (1 - hours.map(hour_freq)).values
            out.loc[idx, "weekday_rarity"] = (1 - weekdays.map(weekday_freq)).values

        out["time_since_prev_txn_seconds"] = out["time_since_prev_txn_seconds"].replace([np.inf, -np.inf], np.nan).fillna(9e9)
        return out

    def build_geo_features(self, transactions_df: pd.DataFrame, locations_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        _ = locations_df, users_df
        df = transactions_df.copy()
        out = df[["transaction_id", "sender_id", "timestamp", "location"]].copy()

        out["distance_from_residence_km"] = np.nan
        out["distance_from_latest_gps_km"] = np.nan
        out["geo_novelty"] = 0.0

        for idx, row in df.iterrows():
            tx_lat = row.get("city_lat")
            tx_lng = row.get("city_lng")
            if pd.isna(tx_lat) or pd.isna(tx_lng):
                continue

            rlat = row.get("sender_residence_lat")
            rlng = row.get("sender_residence_lng")
            if not pd.isna(rlat) and not pd.isna(rlng):
                out.at[idx, "distance_from_residence_km"] = haversine_km(float(rlat), float(rlng), float(tx_lat), float(tx_lng))

            glat = row.get("latest_gps_lat")
            glng = row.get("latest_gps_lng")
            if not pd.isna(glat) and not pd.isna(glng):
                out.at[idx, "distance_from_latest_gps_km"] = haversine_km(float(glat), float(glng), float(tx_lat), float(tx_lng))

        out = out.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
        seen: dict[str, set[str]] = {}
        novelty = []
        for _, row in out.iterrows():
            sender = str(row.get("sender_id"))
            city = str(row.get("location") or "")
            sender_seen = seen.setdefault(sender, set())
            is_new = float(city not in sender_seen and city != "")
            novelty.append(is_new)
            if city:
                sender_seen.add(city)
        out["geo_novelty"] = novelty
        return out

    def build_communication_features(self, transactions_df: pd.DataFrame, sms_df: pd.DataFrame, mails_df: pd.DataFrame) -> pd.DataFrame:
        tx = transactions_df[["transaction_id", "timestamp", "sender_first_name", "recipient_first_name"]].copy()
        tx["suspicious_communication_window_score"] = 0.0
        tx["comm_events_past_7d"] = 0.0

        comm_rows = []
        for _, row in sms_df.iterrows():
            text = str(row.get("message_text") or "")
            comm_rows.append(
                {
                    "timestamp": row.get("timestamp"),
                    "text": text,
                    "base_score": self._communication_text_score(text),
                }
            )
        for _, row in mails_df.iterrows():
            text = str(row.get("body_text") or "") + " " + str(row.get("subject") or "")
            comm_rows.append(
                {
                    "timestamp": row.get("timestamp"),
                    "text": text,
                    "base_score": self._communication_text_score(text),
                }
            )

        comm_df = pd.DataFrame(comm_rows)
        if comm_df.empty:
            return tx

        comm_df["timestamp"] = pd.to_datetime(comm_df["timestamp"], errors="coerce")
        comm_df = comm_df.dropna(subset=["timestamp"]).copy()
        comm_df["text_l"] = comm_df["text"].str.lower()

        for idx, row in tx.iterrows():
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

            mask = pd.Series(False, index=comm_df.index)
            for name in names:
                mask = mask | comm_df["text_l"].str.contains(re.escape(name), regex=True)

            scoped = comm_df[mask & (comm_df["timestamp"] <= ts) & (comm_df["timestamp"] >= ts - pd.Timedelta(days=30))]
            if scoped.empty:
                continue

            hours = (ts - scoped["timestamp"]).dt.total_seconds() / 3600.0
            decay = np.exp(-hours / 48.0)
            weighted = scoped["base_score"].values * decay.values
            tx.at[idx, "suspicious_communication_window_score"] = float(np.clip(weighted.max(), 0.0, 1.0))
            tx.at[idx, "comm_events_past_7d"] = float((hours <= 24 * 7).sum())

        return tx

    def build_reference_delta_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        if self.baseline_store is None:
            out = transactions_df[["transaction_id"]].copy()
            out["ref_sender_amount_robust_z"] = 0.0
            out["ref_recipient_amount_robust_z"] = 0.0
            out["pair_seen_in_reference"] = 0
            out["payment_method_seen_by_sender_ref"] = 0
            out["transaction_type_seen_by_sender_ref"] = 0
            out["reference_hour_rarity"] = 1.0
            out["reference_weekday_rarity"] = 1.0
            out["reference_geo_novelty"] = 1.0
            out["unseen_transaction_type_indicator"] = 1
            out["unseen_payment_method_indicator"] = 1
            out["unseen_location_pattern_indicator"] = 1
            return out
        return self.baseline_store.transform_target(transactions_df)

    def build_novelty_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        out = features_df[["transaction_id"]].copy()
        numeric_cols = [
            c
            for c in features_df.columns
            if c not in {"transaction_id", "timestamp", "sender_id", "recipient_id", "description", "location"}
            and pd.api.types.is_numeric_dtype(features_df[c])
        ]

        if not numeric_cols or len(features_df) < 8:
            out["iso_anomaly_score"] = 0.0
            out["lof_anomaly_score"] = 0.0
            out["novelty_score"] = 0.0
            return out

        x = features_df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True)).fillna(0.0)

        contamination = min(max(0.01, 1.0 / math.sqrt(max(len(x), 2))), 0.15)
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        iso.fit(x)
        iso_score = minmax(pd.Series(-iso.decision_function(x), index=features_df.index))

        n_neighbors = min(25, max(5, len(x) // 10))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit(x)
        lof_score = minmax(pd.Series(-lof.negative_outlier_factor_, index=features_df.index))

        out["iso_anomaly_score"] = iso_score.values
        out["lof_anomaly_score"] = lof_score.values
        out["novelty_score"] = (out["iso_anomaly_score"] + out["lof_anomaly_score"]) / 2.0
        return out

    def build_all(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        sms_df: pd.DataFrame,
        mails_df: pd.DataFrame,
    ) -> pd.DataFrame:
        base = self.build_transaction_features(transactions_df)
        temporal = self.build_temporal_features(base)
        geo = self.build_geo_features(transactions_df, locations_df, users_df)
        comm = self.build_communication_features(transactions_df, sms_df, mails_df)
        ref_delta = self.build_reference_delta_features(transactions_df)

        merged = base.merge(temporal, on=["transaction_id", "sender_id", "timestamp"], how="left")
        merged = merged.merge(
            geo[["transaction_id", "distance_from_residence_km", "distance_from_latest_gps_km", "geo_novelty"]],
            on="transaction_id",
            how="left",
        )
        merged = merged.merge(
            comm[["transaction_id", "suspicious_communication_window_score", "comm_events_past_7d"]],
            on="transaction_id",
            how="left",
        )
        merged = merged.merge(ref_delta, on="transaction_id", how="left")

        novelty = self.build_novelty_features(merged)
        merged = merged.merge(novelty, on="transaction_id", how="left")
        return merged

    @staticmethod
    def _communication_text_score(text: str) -> float:
        t = (text or "").lower()
        keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in t)
        has_url = 1 if ("http://" in t or "https://" in t) else 0
        suspicious_brand_typos = sum(1 for kw in ["paypa1", "amaz0n", "netfl1x", "secure"] if kw in t)
        raw = 0.25 * keyword_hits + 0.2 * has_url + 0.35 * suspicious_brand_typos
        return float(np.clip(raw, 0.0, 1.0))
