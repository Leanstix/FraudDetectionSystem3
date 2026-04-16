from __future__ import annotations

from datetime import datetime, timezone
import mimetypes
from pathlib import Path
import re


def infer_speaker_from_filename(file_name: str) -> str | None:
    stem = Path(file_name).stem
    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\d{4}[-_]?\d{2}[-_]?\d{2}.*$", "", cleaned).strip()
    if not cleaned:
        return None
    words = [w for w in cleaned.split() if len(w) > 1 and not w.isdigit()]
    if not words:
        return None
    return " ".join(words[:3])


def infer_timestamp_from_filename(file_name: str) -> str | None:
    stem = Path(file_name).stem
    patterns = [
        r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)[T_\- ]?([0-2]\d)?[:_\-]?([0-5]\d)?[:_\-]?([0-5]\d)?",
        r"(\d{4})[-_](\d{2})[-_](\d{2})",
    ]
    for pattern in patterns:
        m = re.search(pattern, stem)
        if not m:
            continue
        parts = [p for p in m.groups() if p is not None]
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3]) if len(parts) > 3 else 0
            minute = int(parts[4]) if len(parts) > 4 else 0
            second = int(parts[5]) if len(parts) > 5 else 0
            dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            continue
    return None


def get_audio_metadata(file_path: str) -> dict:
    path = Path(file_path)
    metadata: dict = {
        "duration_seconds": None,
        "mime_type": mimetypes.guess_type(path.name)[0] or "audio/mpeg",
        "size_bytes": path.stat().st_size if path.exists() else None,
    }

    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path))
        if audio is not None and hasattr(audio, "info") and hasattr(audio.info, "length"):
            metadata["duration_seconds"] = float(audio.info.length)
    except Exception:
        return metadata

    return metadata
