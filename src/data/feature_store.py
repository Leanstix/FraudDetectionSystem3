from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.constants import SUSPICIOUS_KEYWORDS
from src.utils.geo import haversine_km
from src.utils.math import minmax, robust_zscore


def _safe_robust_z(value: float, median: float, mad: float, std: float) -> float:
    if pd.isna(value):
        return 0.0
    if pd.notna(mad) and mad > 0:
        return float(0.6745 * (value - median) / mad)
    if pd.notna(std) and std > 0:
        return float((value - median) / std)
    return 0.0


class FeatureStore:
    def __init__(self):
        self.seed = 42

    def build_transaction_features(
        self,
        reference_transactions_df: pd.DataFrame,
        target_transactions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ref = reference_transactions_df.copy()
        tgt = target_transactions_df.copy().sort_values("timestamp").reset_index(drop=True)

        # Target-local robust z features.
        tgt["amount_robust_z_sender"] = tgt.groupby("sender_id", dropna=False)["amount"].transform(
            lambda s: robust_zscore(s.astype(float).fillna(0.0))
        )
        tgt["amount_robust_z_recipient"] = tgt.groupby("recipient_id", dropna=False)["amount"].transform(
            lambda s: robust_zscore(s.astype(float).fillna(0.0))
        )

        # Reference baseline robust z features.
        sender_profile = (
            ref.groupby("sender_id", dropna=False)["amount"]
            .agg(sender_median="median", sender_std="std")
            .reset_index()
        )
        sender_mad = (
            ref.groupby("sender_id", dropna=False)["amount"]
            .apply(lambda s: (s - s.median()).abs().median())
            .rename("sender_mad")
            .reset_index()
        )
        sender_profile = sender_profile.merge(sender_mad, on="sender_id", how="left")

        recipient_profile = (
            ref.groupby("recipient_id", dropna=False)["amount"]
            .agg(recipient_median="median", recipient_std="std")
            .reset_index()
        )
        recipient_mad = (
            ref.groupby("recipient_id", dropna=False)["amount"]
            .apply(lambda s: (s - s.median()).abs().median())
            .rename("recipient_mad")
            .reset_index()
        )
        recipient_profile = recipient_profile.merge(recipient_mad, on="recipient_id", how="left")

        tgt = tgt.merge(sender_profile, on="sender_id", how="left")
        tgt = tgt.merge(recipient_profile, on="recipient_id", how="left")

        tgt["ref_sender_amount_robust_z"] = tgt.apply(
            lambda r: _safe_robust_z(r.get("amount"), r.get("sender_median"), r.get("sender_mad"), r.get("sender_std")),
            axis=1,
        )
        tgt["ref_recipient_amount_robust_z"] = tgt.apply(
            lambda r: _safe_robust_z(
                r.get("amount"),
                r.get("recipient_median"),
                r.get("recipient_mad"),
                r.get("recipient_std"),
            ),
            axis=1,
        )

        pair_ref = set((ref["sender_id"].fillna(""), ref["recipient_id"].fillna("")))
        tgt["pair_key"] = list(zip(tgt["sender_id"].fillna(""), tgt["recipient_id"].fillna("")))
        tgt["pair_seen_in_reference"] = tgt["pair_key"].map(lambda p: 1 if p in pair_ref else 0)
        tgt["new_sender_recipient_pair"] = 1 - tgt["pair_seen_in_reference"]

        sender_method_ref = (
            ref.groupby("sender_id", dropna=False)["payment_method"]
            .apply(lambda s: set(v for v in s.fillna("").astype(str) if v))
            .to_dict()
        )
        sender_type_ref = (
            ref.groupby("sender_id", dropna=False)["transaction_type"]
            .apply(lambda s: set(v for v in s.fillna("").astype(str) if v))
            .to_dict()
        )

        tgt["payment_method_seen_by_sender_ref"] = tgt.apply(
            lambda r: 1
            if str(r.get("payment_method") or "") in sender_method_ref.get(r.get("sender_id"), set())
            else 0,
            axis=1,
        )
        tgt["transaction_type_seen_by_sender_ref"] = tgt.apply(
            lambda r: 1
            if str(r.get("transaction_type") or "") in sender_type_ref.get(r.get("sender_id"), set())
            else 0,
            axis=1,
        )
        tgt["unseen_transaction_type_indicator"] = 1 - tgt["transaction_type_seen_by_sender_ref"]
        tgt["unseen_payment_method_indicator"] = 1 - tgt["payment_method_seen_by_sender_ref"]

        desc = tgt["description"].fillna("").astype(str)
        kw_pattern = "|".join(re.escape(k) for k in SUSPICIOUS_KEYWORDS)
        tgt["description_present"] = desc.str.strip().ne("").astype(int)
        tgt["description_length"] = desc.str.len()
        tgt["description_keyword_hits"] = desc.str.lower().str.count(kw_pattern)

        return tgt

    def build_temporal_features(
        self,
        reference_transactions_df: pd.DataFrame,
        target_transactions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        ref = reference_transactions_df.copy()
        tgt = target_transactions_df.copy()

        out = tgt[["transaction_id", "sender_id", "timestamp"]].copy().sort_values(["sender_id", "timestamp"])
        out["txn_count_past_1h"] = 0.0
        out["txn_count_past_24h"] = 0.0
        out["txn_count_past_7d"] = 0.0
        out["time_since_prev_txn_seconds"] = 9e9
        out["burst_count_10min"] = 0.0

        combined = pd.concat(
            [
                ref[["sender_id", "timestamp"]].assign(is_target=0),
                tgt[["sender_id", "timestamp"]].assign(is_target=1),
            ],
            ignore_index=True,
        ).sort_values(["sender_id", "timestamp"])

        for sender, grp in combined.groupby("sender_id", dropna=False):
            g = grp.set_index("timestamp").sort_index()
            c1h = (g["is_target"].rolling("1h").count() - 1).clip(lower=0)
            c24h = (g["is_target"].rolling("24h").count() - 1).clip(lower=0)
            c7d = (g["is_target"].rolling("7D").count() - 1).clip(lower=0)
            burst = (g["is_target"].rolling("10min").count() - 1).clip(lower=0)

            # Assign only to target timestamps for this sender.
            tgt_rows = tgt[tgt["sender_id"].eq(sender)].sort_values("timestamp")
            for idx, row in tgt_rows.iterrows():
                ts = row["timestamp"]
                if pd.isna(ts):
                    continue
                mask = (combined["sender_id"].eq(sender)) & (combined["timestamp"].eq(ts))
                subset = combined[mask]
                if subset.empty:
                    continue
                loc = subset.index[-1]
                out_idx = out.index[out["transaction_id"].eq(row["transaction_id"])]
                if len(out_idx) == 0:
                    continue
                out.loc[out_idx, "txn_count_past_1h"] = float(c1h.iloc[g.index.get_indexer([ts], method="pad")[0]]) if not c1h.empty else 0.0
                out.loc[out_idx, "txn_count_past_24h"] = float(c24h.iloc[g.index.get_indexer([ts], method="pad")[0]]) if not c24h.empty else 0.0
                out.loc[out_idx, "txn_count_past_7d"] = float(c7d.iloc[g.index.get_indexer([ts], method="pad")[0]]) if not c7d.empty else 0.0
                out.loc[out_idx, "burst_count_10min"] = float(burst.iloc[g.index.get_indexer([ts], method="pad")[0]]) if not burst.empty else 0.0

        # Time since previous transaction including reference history.
        for sender, grp in combined.groupby("sender_id", dropna=False):
            grp = grp.sort_values("timestamp")
            grp["delta"] = grp["timestamp"].diff().dt.total_seconds().fillna(9e9)
            for _, row in grp[grp["is_target"] == 1].iterrows():
                out_idx = out.index[out["sender_id"].eq(sender) & out["timestamp"].eq(row["timestamp"])]
                out.loc[out_idx, "time_since_prev_txn_seconds"] = float(max(row["delta"], 0.0))

        ref_hour_freq = ref["timestamp"].dt.hour.value_counts(normalize=True).to_dict()
        ref_weekday_freq = ref["timestamp"].dt.weekday.value_counts(normalize=True).to_dict()

        out["hour_rarity"] = 1 - out["timestamp"].dt.hour.map(out["timestamp"].dt.hour.value_counts(normalize=True)).fillna(0)
        out["weekday_rarity"] = 1 - out["timestamp"].dt.weekday.map(out["timestamp"].dt.weekday.value_counts(normalize=True)).fillna(0)
        out["reference_hour_rarity"] = 1 - out["timestamp"].dt.hour.map(ref_hour_freq).fillna(0)
        out["reference_weekday_rarity"] = 1 - out["timestamp"].dt.weekday.map(ref_weekday_freq).fillna(0)

        return out

    def build_geo_features(self, target_transactions_df: pd.DataFrame, locations_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        _ = users_df
        df = target_transactions_df.copy()
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
            if pd.notna(rlat) and pd.notna(rlng):
                out.at[idx, "distance_from_residence_km"] = haversine_km(float(rlat), float(rlng), float(tx_lat), float(tx_lng))

            glat = row.get("latest_gps_lat")
            glng = row.get("latest_gps_lng")
            if pd.notna(glat) and pd.notna(glng):
                out.at[idx, "distance_from_latest_gps_km"] = haversine_km(float(glat), float(glng), float(tx_lat), float(tx_lng))

        out = out.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)
        seen: dict[str, set[str]] = {}
        novelty: list[float] = []
        for _, row in out.iterrows():
            sender = str(row.get("sender_id"))
            city = str(row.get("location") or "")
            sender_seen = seen.setdefault(sender, set())
            is_new = float(city not in sender_seen and city != "")
            novelty.append(is_new)
            if city:
                sender_seen.add(city)
        out["geo_novelty"] = novelty

        # From geo traces: simple novelty indicator based on sender signature coverage.
        sig_coverage = 0.0
        if not locations_df.empty and "biotag_signature" in locations_df.columns and "sender_user_signature" in df.columns:
            known = set(locations_df["biotag_signature"].dropna().astype(str))
            out["unseen_location_pattern_indicator"] = df["sender_user_signature"].astype(str).map(lambda s: 0 if s in known else 1)
            sig_coverage = float((df["sender_user_signature"].astype(str).isin(known)).mean())
        else:
            out["unseen_location_pattern_indicator"] = 1

        out["reference_geo_novelty"] = out["geo_novelty"] * (1.0 - sig_coverage)
        return out

    def build_communication_features(self, target_transactions_df: pd.DataFrame, sms_df: pd.DataFrame, mails_df: pd.DataFrame) -> pd.DataFrame:
        tx = target_transactions_df[["transaction_id", "timestamp", "sender_first_name", "recipient_first_name"]].copy()
        tx["suspicious_communication_window_score"] = 0.0
        tx["comm_events_past_7d"] = 0.0

        comm_rows: list[dict] = []
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
            decay = np.exp(-hours / 72.0)
            weighted = scoped["base_score"].values * decay.values
            tx.at[idx, "suspicious_communication_window_score"] = float(np.clip(weighted.max(), 0.0, 1.0))
            tx.at[idx, "comm_events_past_7d"] = float((hours <= 24 * 7).sum())

        return tx

    def build_audio_features(self, target_transactions_df: pd.DataFrame, audio_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        tx = target_transactions_df[["transaction_id", "timestamp", "sender_user_idx", "sender_full_name"]].copy()
        tx["audio_activity_score"] = 0.0
        tx["audio_proximity_score"] = 0.0
        tx["recent_audio_event_by_linked_user"] = 0.0

        if audio_df.empty:
            return tx

        users = users_df.copy()
        users["full_name_norm"] = users.get("user_full_name", pd.Series(index=users.index, dtype=str)).fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

        audio = audio_df.copy()
        audio["speaker_norm"] = audio.get("speaker_norm", pd.Series(index=audio.index, dtype=str)).fillna("")
        audio["duration_seconds"] = pd.to_numeric(audio.get("duration_seconds", pd.Series(index=audio.index, dtype=float)), errors="coerce")
        audio["metadata_risk"] = minmax(
            audio["duration_seconds"].fillna(0.0).clip(lower=0.0, upper=7200.0)
            + minmax(audio.get("size_bytes", pd.Series(index=audio.index, dtype=float)).fillna(0.0))
        )

        # Speaker-to-user linkage.
        user_map: dict[str, int] = {}
        for _, row in users.iterrows():
            full = str(row.get("full_name_norm") or "").strip()
            first = str(row.get("first_name") or "").strip().lower()
            last = str(row.get("last_name") or "").strip().lower()
            uid = int(row.get("user_idx"))
            if full:
                user_map[full] = uid
            if first and last:
                user_map[f"{first} {last}"] = uid
            if last:
                user_map[last] = uid

        audio["speaker_user_idx"] = audio["speaker_norm"].map(user_map)

        for idx, row in tx.iterrows():
            ts = row.get("timestamp")
            sender_idx = row.get("sender_user_idx")
            if pd.isna(ts):
                continue

            linked = pd.DataFrame()
            if pd.notna(sender_idx):
                linked = audio[audio["speaker_user_idx"].eq(sender_idx)]
            if linked.empty:
                sender_name = str(row.get("sender_full_name") or "").strip().lower()
                if sender_name:
                    linked = audio[audio["speaker_norm"].str.contains(re.escape(sender_name), regex=True, na=False)]

            if linked.empty:
                continue

            prior = linked[linked["timestamp"] <= ts]
            if prior.empty:
                continue

            hours = (ts - prior["timestamp"]).dt.total_seconds() / 3600.0
            decay = np.exp(-hours / 72.0)
            decayed = prior["metadata_risk"].fillna(0.0).values * decay.values

            tx.at[idx, "audio_activity_score"] = float(np.clip(decayed.max(), 0.0, 1.0))
            tx.at[idx, "audio_proximity_score"] = float(np.clip(np.exp(-hours.min() / 24.0), 0.0, 1.0))
            tx.at[idx, "recent_audio_event_by_linked_user"] = 1.0 if float(hours.min()) <= 24.0 else 0.0

        return tx

    def build_novelty_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        out = features_df[["transaction_id"]].copy()
        numeric_cols = [
            c
            for c in features_df.columns
            if c
            not in {
                "transaction_id",
                "timestamp",
                "sender_id",
                "recipient_id",
                "description",
                "location",
                "sender_first_name",
                "recipient_first_name",
                "sender_full_name",
                "recipient_full_name",
            }
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
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=self.seed)
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
        reference_transactions_df: pd.DataFrame,
        target_transactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        sms_df: pd.DataFrame,
        mails_df: pd.DataFrame,
        audio_df: pd.DataFrame,
    ) -> pd.DataFrame:
        base = self.build_transaction_features(reference_transactions_df, target_transactions_df)
        temporal = self.build_temporal_features(reference_transactions_df, target_transactions_df)
        geo = self.build_geo_features(target_transactions_df, locations_df, users_df)
        comm = self.build_communication_features(target_transactions_df, sms_df, mails_df)
        audio = self.build_audio_features(target_transactions_df, audio_df, users_df)

        merged = base.merge(temporal, on=["transaction_id", "sender_id", "timestamp"], how="left")
        merged = merged.merge(
            geo[
                [
                    "transaction_id",
                    "distance_from_residence_km",
                    "distance_from_latest_gps_km",
                    "geo_novelty",
                    "reference_geo_novelty",
                    "unseen_location_pattern_indicator",
                ]
            ],
            on="transaction_id",
            how="left",
        )
        merged = merged.merge(
            comm[["transaction_id", "suspicious_communication_window_score", "comm_events_past_7d"]],
            on="transaction_id",
            how="left",
        )
        merged = merged.merge(
            audio[["transaction_id", "audio_activity_score", "audio_proximity_score", "recent_audio_event_by_linked_user"]],
            on="transaction_id",
            how="left",
        )

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
