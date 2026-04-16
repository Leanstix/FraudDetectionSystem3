from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.data.schemas import validate_transactions_schema
from src.utils.audio import (
    get_audio_metadata,
    infer_speaker_from_filename,
    infer_timestamp_from_filename,
)
from src.utils.text import html_to_text, parse_mail_headers
from src.utils.time import parse_datetime_series


def _token_from_name(value: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", value or "").upper()
    consonants = "".join(ch for ch in letters if ch not in "AEIOU")
    vowels = "".join(ch for ch in letters if ch in "AEIOU")
    token = (consonants + vowels)[:4]
    return token.ljust(4, "X")


def _extract_id_signature(identifier: str) -> str | None:
    if not isinstance(identifier, str):
        return None
    parts = identifier.split("-")
    if len(parts) < 2:
        return None
    if len(parts[0]) == 4 and len(parts[1]) == 4:
        return f"{parts[0].upper()}-{parts[1].upper()}"
    return None


def _parse_sms_messages(raw_sms: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"From:\s*(.*?)\nTo:\s*(.*?)\nDate:\s*(.*?)\nMessage:\s*(.*?)(?=\nFrom: |\n=== END CONVERSATION ===|\Z)",
        flags=re.DOTALL,
    )
    rows: list[dict[str, Any]] = []
    for m in pattern.finditer(raw_sms or ""):
        rows.append(
            {
                "sender": (m.group(1) or "").strip(),
                "recipient": (m.group(2) or "").strip(),
                "timestamp": (m.group(3) or "").strip(),
                "message_text": (m.group(4) or "").strip(),
            }
        )
    if rows:
        return rows

    lines = (raw_sms or "").splitlines()
    sender = recipient = timestamp = None
    message_parts: list[str] = []
    for line in lines:
        if line.startswith("From:"):
            sender = line.split(":", 1)[1].strip()
        elif line.startswith("To:"):
            recipient = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            timestamp = line.split(":", 1)[1].strip()
        elif line.startswith("Message:"):
            message_parts.append(line.split(":", 1)[1].strip())
        elif message_parts:
            message_parts.append(line.strip())

    if sender or recipient or timestamp or message_parts:
        rows.append(
            {
                "sender": sender,
                "recipient": recipient,
                "timestamp": timestamp,
                "message_text": " ".join(part for part in message_parts if part),
            }
        )
    return rows


class Normalizer:
    def normalize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        validate_transactions_schema(out)

        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})

        for col in ["amount", "balance_after"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out["timestamp"] = parse_datetime_series(out["timestamp"])
        out = out.sort_values("timestamp").reset_index(drop=True)
        return out

    def normalize_users(self, rows: list[dict]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            residence = row.get("residence") or {}
            first = str(row.get("first_name", "")).strip()
            last = str(row.get("last_name", "")).strip()
            records.append(
                {
                    "user_idx": idx,
                    "first_name": first or pd.NA,
                    "last_name": last or pd.NA,
                    "birth_year": pd.to_numeric(row.get("birth_year"), errors="coerce"),
                    "salary": pd.to_numeric(row.get("salary"), errors="coerce"),
                    "job": str(row.get("job", "")).strip() or pd.NA,
                    "iban": str(row.get("iban", "")).strip() or pd.NA,
                    "residence_city": str(residence.get("city", "")).strip() or pd.NA,
                    "residence_lat": pd.to_numeric(residence.get("lat"), errors="coerce"),
                    "residence_lng": pd.to_numeric(residence.get("lng"), errors="coerce"),
                    "description": str(row.get("description", "")).strip() or pd.NA,
                }
            )

        out = pd.DataFrame.from_records(records)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "user_idx",
                    "first_name",
                    "last_name",
                    "birth_year",
                    "salary",
                    "job",
                    "iban",
                    "residence_city",
                    "residence_lat",
                    "residence_lng",
                    "description",
                    "user_full_name",
                    "name_token_last",
                    "name_token_first",
                    "user_signature",
                    "city_token",
                ]
            )

        out["user_full_name"] = (out["first_name"].fillna("") + " " + out["last_name"].fillna(" ")).str.strip()
        out["name_token_last"] = out["last_name"].fillna("").map(_token_from_name)
        out["name_token_first"] = out["first_name"].fillna("").map(_token_from_name)
        out["user_signature"] = out["name_token_last"] + "-" + out["name_token_first"]
        out["city_token"] = out["residence_city"].fillna("").astype(str).str[:3].str.upper()
        return out

    def normalize_locations(self, rows: list[dict]) -> pd.DataFrame:
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["biotag", "timestamp", "lat", "lng", "city", "biotag_signature"])

        out["biotag"] = out.get("biotag", pd.Series(dtype=str)).astype(str).str.strip()
        out["timestamp"] = parse_datetime_series(out.get("timestamp", pd.Series(dtype=str)))
        out["lat"] = pd.to_numeric(out.get("lat", pd.Series(dtype=float)), errors="coerce")
        out["lng"] = pd.to_numeric(out.get("lng", pd.Series(dtype=float)), errors="coerce")
        out["city"] = out.get("city", pd.Series(dtype=str)).astype(str).str.strip()
        out["biotag_signature"] = out["biotag"].map(_extract_id_signature)
        out = out.sort_values("timestamp").reset_index(drop=True)
        return out

    def normalize_sms(self, rows: list[dict]) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            raw_sms = str(row.get("sms", ""))
            parsed = _parse_sms_messages(raw_sms)
            for msg_idx, msg in enumerate(parsed):
                normalized.append(
                    {
                        "thread_id": f"sms_{idx}",
                        "message_id": f"sms_{idx}_{msg_idx}",
                        "sender": msg.get("sender"),
                        "recipient": msg.get("recipient"),
                        "timestamp": msg.get("timestamp"),
                        "message_text": msg.get("message_text", ""),
                        "raw_text": raw_sms,
                    }
                )

        out = pd.DataFrame.from_records(normalized)
        if out.empty:
            return pd.DataFrame(columns=["thread_id", "sender", "recipient", "timestamp", "message_text", "raw_text"])

        out["timestamp"] = parse_datetime_series(out["timestamp"])
        out["message_text"] = out["message_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    def normalize_mails(self, rows: list[dict]) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            raw_mail = str(row.get("mail", ""))
            headers = parse_mail_headers(raw_mail)
            body = raw_mail.split("\n\n", 1)[1] if "\n\n" in raw_mail else raw_mail
            plain = html_to_text(body)
            normalized.append(
                {
                    "thread_id": f"mail_{idx}",
                    "sender": headers.get("from"),
                    "recipient": headers.get("to"),
                    "subject": headers.get("subject"),
                    "timestamp": headers.get("date"),
                    "body_text": plain,
                    "raw_text": raw_mail,
                }
            )

        out = pd.DataFrame.from_records(normalized)
        if out.empty:
            return pd.DataFrame(columns=["thread_id", "sender", "recipient", "subject", "timestamp", "body_text", "raw_text"])

        out["timestamp"] = parse_datetime_series(out["timestamp"])
        out["body_text"] = out["body_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    def normalize_audio(self, rows: list[dict]) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            file_name = str(row.get("file_name") or "").strip()
            file_path = str(row.get("file_path") or "").strip()

            inferred_speaker = infer_speaker_from_filename(file_name)
            inferred_ts = infer_timestamp_from_filename(file_name)
            metadata = get_audio_metadata(file_path) if file_path else {"duration_seconds": None, "mime_type": None, "size_bytes": None}

            normalized.append(
                {
                    "audio_id": str(row.get("audio_id") or f"audio_{idx}"),
                    "file_path": file_path,
                    "file_name": file_name,
                    "relative_path": str(row.get("relative_path") or file_name),
                    "inferred_speaker": inferred_speaker,
                    "inferred_timestamp": inferred_ts,
                    "duration_seconds": metadata.get("duration_seconds"),
                    "mime_type": metadata.get("mime_type"),
                    "size_bytes": metadata.get("size_bytes", row.get("size_bytes")),
                    "metadata": metadata,
                }
            )

        out = pd.DataFrame.from_records(normalized)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "audio_id",
                    "file_path",
                    "file_name",
                    "relative_path",
                    "inferred_speaker",
                    "inferred_timestamp",
                    "duration_seconds",
                    "mime_type",
                    "size_bytes",
                    "timestamp",
                ]
            )

        out["timestamp"] = parse_datetime_series(out["inferred_timestamp"])
        out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce")
        out["size_bytes"] = pd.to_numeric(out["size_bytes"], errors="coerce")
        out["speaker_norm"] = out["inferred_speaker"].fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        return out
