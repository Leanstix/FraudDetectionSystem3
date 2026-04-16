from __future__ import annotations

import pandas as pd

from src.config import Settings
from src.llm.audio_reasoner import AudioReasoner
from src.llm.client import LLMClient
from src.tracing import TracingManager


def test_metadata_only_audio_path_works(tmp_path):
    settings = Settings.from_env_and_file()
    settings.audio_transcription_enabled = False
    settings.llm_enabled = False

    tracing = TracingManager(settings)
    llm_client = LLMClient(settings, tracing)
    reasoner = AudioReasoner(llm_client, settings)

    fake_audio = tmp_path / "Agent Smith_2026-04-01T011500.mp3"
    fake_audio.write_bytes(b"ID3\x00\x00\x00")

    record = {
        "audio_id": "audio_1",
        "file_path": str(fake_audio),
        "file_name": fake_audio.name,
        "inferred_speaker": "Agent Smith",
        "duration_seconds": 12.0,
        "timestamp": pd.Timestamp("2026-04-01T01:15:00Z"),
    }

    scored = reasoner.metadata_only_audio_score(record)
    assert 0.0 <= float(scored["risk_score"]) <= 1.0
    assert "reason" in scored

    transcript = reasoner.transcribe_if_available(str(fake_audio))
    assert transcript["ok"] is False
