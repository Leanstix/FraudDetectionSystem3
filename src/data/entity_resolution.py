from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _extract_id_signature(identifier: str) -> str | None:
    if not isinstance(identifier, str):
        return None
    parts = identifier.split("-")
    if len(parts) < 2:
        return None
    if len(parts[0]) == 4 and len(parts[1]) == 4:
        return f"{parts[0].upper()}-{parts[1].upper()}"
    return None


class EntityResolver:
    def build_user_index(self, users_df: pd.DataFrame) -> pd.DataFrame:
        out = users_df.copy().reset_index(drop=True)
        out["user_idx"] = out.get("user_idx", out.index)
        out["iban"] = out.get("iban", pd.Series(index=out.index, dtype=str)).astype(str).str.strip()
        out["user_signature"] = out.get("user_signature", pd.Series(index=out.index, dtype=str)).astype(str).str.upper()
        out["full_name_norm"] = out.get("user_full_name", pd.Series(index=out.index, dtype=str)).fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    def link_transactions_to_users(self, transactions_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        tx = transactions_df.copy()
        users = self.build_user_index(users_df)

        sig_to_user = users.dropna(subset=["user_signature"]).drop_duplicates("user_signature").set_index("user_signature")
        iban_to_user = users.dropna(subset=["iban"]).drop_duplicates("iban").set_index("iban")

        tx["sender_signature"] = tx["sender_id"].astype(str).map(_extract_id_signature)
        tx["recipient_signature"] = tx["recipient_id"].astype(str).map(_extract_id_signature)

        tx["sender_user_idx_sig"] = tx["sender_signature"].map(sig_to_user["user_idx"] if not sig_to_user.empty else pd.Series(dtype=float))
        tx["recipient_user_idx_sig"] = tx["recipient_signature"].map(sig_to_user["user_idx"] if not sig_to_user.empty else pd.Series(dtype=float))

        tx["sender_user_idx_iban"] = tx["sender_iban"].astype(str).str.strip().map(iban_to_user["user_idx"] if not iban_to_user.empty else pd.Series(dtype=float))
        tx["recipient_user_idx_iban"] = tx["recipient_iban"].astype(str).str.strip().map(iban_to_user["user_idx"] if not iban_to_user.empty else pd.Series(dtype=float))

        tx["sender_user_idx"] = tx["sender_user_idx_iban"].combine_first(tx["sender_user_idx_sig"])
        tx["recipient_user_idx"] = tx["recipient_user_idx_iban"].combine_first(tx["recipient_user_idx_sig"])

        tx["sender_link_score"] = np.where(
            tx["sender_user_idx_iban"].notna(),
            0.95,
            np.where(tx["sender_user_idx_sig"].notna(), 0.7, 0.0),
        )
        tx["recipient_link_score"] = np.where(
            tx["recipient_user_idx_iban"].notna(),
            0.95,
            np.where(tx["recipient_user_idx_sig"].notna(), 0.7, 0.0),
        )

        user_cols = [
            "user_idx",
            "first_name",
            "last_name",
            "user_full_name",
            "full_name_norm",
            "residence_city",
            "residence_lat",
            "residence_lng",
            "salary",
            "job",
            "iban",
            "user_signature",
        ]
        sender_users = users[user_cols].add_prefix("sender_")
        recipient_users = users[user_cols].add_prefix("recipient_")

        tx = tx.merge(sender_users, left_on="sender_user_idx", right_on="sender_user_idx", how="left")
        tx = tx.merge(recipient_users, left_on="recipient_user_idx", right_on="recipient_user_idx", how="left")
        return tx

    def attach_location_context(self, transactions_df: pd.DataFrame, locations_df: pd.DataFrame) -> pd.DataFrame:
        tx = transactions_df.copy()
        loc = locations_df.copy()

        city_centroids = (
            loc.dropna(subset=["city", "lat", "lng"])
            .groupby("city", as_index=False)[["lat", "lng"]]
            .mean()
            .rename(columns={"lat": "city_lat", "lng": "city_lng"})
        )
        tx = tx.merge(city_centroids, left_on="location", right_on="city", how="left")
        tx = tx.drop(columns=["city"], errors="ignore")

        tx["latest_gps_lat"] = np.nan
        tx["latest_gps_lng"] = np.nan
        tx["latest_gps_timestamp"] = pd.Series([pd.NaT] * len(tx), dtype="datetime64[ns, UTC]")

        if loc.empty:
            return tx

        loc = loc.dropna(subset=["biotag_signature", "timestamp", "lat", "lng"]).sort_values("timestamp")
        tx = tx.sort_values("timestamp").reset_index(drop=True)

        loc_groups = {k: g for k, g in loc.groupby("biotag_signature")}

        for idx, row in tx.iterrows():
            sig = row.get("sender_user_signature")
            ts = row.get("timestamp")
            if not isinstance(sig, str) or not sig or pd.isna(ts):
                continue
            group = loc_groups.get(sig)
            if group is None or group.empty:
                continue
            prior = group[group["timestamp"] <= ts]
            if prior.empty:
                continue
            latest = prior.iloc[-1]
            tx.at[idx, "latest_gps_lat"] = latest["lat"]
            tx.at[idx, "latest_gps_lng"] = latest["lng"]
            tx.at[idx, "latest_gps_timestamp"] = latest["timestamp"]

        return tx

    def attach_communication_context(self, transactions_df: pd.DataFrame, sms_df: pd.DataFrame, mails_df: pd.DataFrame) -> pd.DataFrame:
        tx = transactions_df.copy()
        sms_text = sms_df.get("message_text", pd.Series(dtype=str)).fillna("").str.lower()
        mail_text = mails_df.get("body_text", pd.Series(dtype=str)).fillna("").str.lower()

        tx["sender_comm_mentions"] = 0
        tx["recipient_comm_mentions"] = 0

        for idx, row in tx.iterrows():
            sender_name = str(row.get("sender_first_name", "") or "").strip().lower()
            recipient_name = str(row.get("recipient_first_name", "") or "").strip().lower()
            if sender_name:
                tx.at[idx, "sender_comm_mentions"] = int(
                    sms_text.str.contains(re.escape(sender_name)).sum()
                    + mail_text.str.contains(re.escape(sender_name)).sum()
                )
            if recipient_name:
                tx.at[idx, "recipient_comm_mentions"] = int(
                    sms_text.str.contains(re.escape(recipient_name)).sum()
                    + mail_text.str.contains(re.escape(recipient_name)).sum()
                )

        return tx

    def attach_audio_context(self, transactions_df: pd.DataFrame, audio_df: pd.DataFrame) -> pd.DataFrame:
        tx = transactions_df.copy()
        out = tx.copy()
        out["audio_link_count"] = 0
        out["audio_recent_event_hours"] = np.nan

        if audio_df.empty:
            return out

        audio = audio_df.copy()
        audio["speaker_norm"] = audio.get("speaker_norm", pd.Series(index=audio.index, dtype=str)).fillna("")

        for idx, row in out.iterrows():
            tx_ts = row.get("timestamp")
            sender_name = str(row.get("sender_full_name") or row.get("sender_user_full_name") or "").strip().lower()
            if pd.isna(tx_ts) or not sender_name:
                continue

            matched = audio[audio["speaker_norm"].str.contains(re.escape(sender_name), regex=True, na=False)]
            if matched.empty:
                continue

            prior = matched[matched["timestamp"] <= tx_ts]
            if prior.empty:
                continue

            out.at[idx, "audio_link_count"] = int(len(prior))
            min_hours = float((tx_ts - prior["timestamp"]).dt.total_seconds().div(3600.0).min())
            out.at[idx, "audio_recent_event_hours"] = min_hours

        return out

    def build_entity_profiles(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        audio_df: pd.DataFrame,
    ) -> dict:
        profiles: dict = {"users": {}, "global": {}}
        tx = transactions_df.copy()

        for _, user in users_df.iterrows():
            uid = int(user["user_idx"])
            sender_rows = tx[tx["sender_user_idx"] == uid]
            recipient_rows = tx[tx["recipient_user_idx"] == uid]
            profiles["users"][uid] = {
                "full_name": user.get("user_full_name"),
                "residence_city": user.get("residence_city"),
                "salary": float(user.get("salary")) if not pd.isna(user.get("salary")) else None,
                "sent_count": int(len(sender_rows)),
                "received_count": int(len(recipient_rows)),
                "sent_amount_mean": float(sender_rows["amount"].mean()) if len(sender_rows) else 0.0,
                "received_amount_mean": float(recipient_rows["amount"].mean()) if len(recipient_rows) else 0.0,
            }

        profiles["global"] = {
            "transaction_count": int(len(tx)),
            "linked_sender_rate": float((tx["sender_user_idx"].notna().mean()) if len(tx) else 0.0),
            "linked_recipient_rate": float((tx["recipient_user_idx"].notna().mean()) if len(tx) else 0.0),
            "location_ping_count": int(len(locations_df)),
            "audio_event_count": int(len(audio_df)),
        }
        return profiles
