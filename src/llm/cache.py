from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LLMCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, model: str, prompt: str) -> str:
        payload = f"{model}::{prompt}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> str | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        path = self.cache_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"value": value}, f, ensure_ascii=False)
