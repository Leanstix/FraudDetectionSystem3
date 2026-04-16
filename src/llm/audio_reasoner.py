from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from src.config import Settings
from src.llm.client import LLMClient
from src.llm.prompts import audio_summary_prompt


class AudioReasoner:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm_client = llm_client
        self.settings = settings

    def transcribe_if_available(self, audio_path: str) -> dict:
        if not self.settings.is_audio_transcription_enabled():
            return {"transcript": "", "engine": "disabled", "ok": False}

        # Lightweight optional path: if whisper package is not installed, skip safely.
        try:
            import whisper
        except Exception:
            return {"transcript": "", "engine": "unavailable", "ok": False}

        try:
            model = whisper.load_model("tiny")
            result = model.transcribe(audio_path)
            text = str(result.get("text") or "").strip()
            return {"transcript": text, "engine": "openai-whisper-tiny", "ok": bool(text)}
        except Exception:
            return {"transcript": "", "engine": "error", "ok": False}

    def analyze_audio_event(self, audio_record: dict, dataset_name: str) -> dict:
        heuristic = self.metadata_only_audio_score(audio_record)
        if heuristic["risk_score"] < 0.45 or not self.llm_client.is_enabled():
            return heuristic

        context = {
            "file_name": audio_record.get("file_name"),
            "speaker": audio_record.get("inferred_speaker"),
            "timestamp": str(audio_record.get("timestamp") or audio_record.get("inferred_timestamp") or ""),
            "duration_seconds": audio_record.get("duration_seconds"),
            "metadata_risk": heuristic.get("risk_score"),
        }
        prompt = audio_summary_prompt(json.dumps(context, ensure_ascii=True))
        raw = self.llm_client.invoke(prompt, dataset_name=dataset_name, task_name="audio_event")
        if not raw:
            return heuristic

        parsed = self._parse_json_like(raw)
        if not parsed:
            return heuristic

        try:
            llm_risk = float(np.clip(float(parsed.get("risk_score", 0.0)), 0.0, 1.0))
        except Exception:
            llm_risk = 0.0

        return {
            "risk_score": float(np.clip(0.65 * heuristic["risk_score"] + 0.35 * llm_risk, 0.0, 1.0)),
            "reason": (heuristic.get("reason", "") + " | llm_audio").strip()[:240],
            "confidence": float(np.clip(float(parsed.get("confidence", 0.5)), 0.0, 1.0)),
        }

    def metadata_only_audio_score(self, audio_record: dict) -> dict:
        duration = float(audio_record.get("duration_seconds") or 0.0)
        duration_norm = float(np.clip(duration / max(self.settings.audio_max_duration_seconds, 1.0), 0.0, 1.0))

        speaker = str(audio_record.get("inferred_speaker") or "").strip().lower()
        speaker_risk = 0.2 if not speaker else 0.0

        file_name = str(audio_record.get("file_name") or "").lower()
        suspicious_hint = 1.0 if any(h in file_name for h in ["urgent", "secure", "bank", "verify", "otp"]) else 0.0

        recency_signal = 0.0
        ts = audio_record.get("timestamp")
        if ts is not None and hasattr(ts, "hour"):
            recency_signal = 0.25 if int(ts.hour) in {0, 1, 2, 3, 4, 5} else 0.0

        score = float(np.clip(0.45 * duration_norm + 0.2 * suspicious_hint + 0.2 * recency_signal + speaker_risk, 0.0, 1.0))
        return {
            "risk_score": score,
            "reason": f"audio_metadata:duration={duration:.1f},speaker={'yes' if speaker else 'missing'}",
            "confidence": 0.6,
        }

    @staticmethod
    def _parse_json_like(raw: str) -> dict | None:
        text = raw.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        blob = match.group(0)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:
                return json.loads(blob.replace("'", '"'))
            except Exception:
                return None
