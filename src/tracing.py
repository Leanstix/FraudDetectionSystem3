from __future__ import annotations

import re
from typing import Any

import ulid

from src.config import Settings


class TracingManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._session_id: str | None = None
        self._langfuse_client: Any = None
        self._callback_handler: Any = None
        self._langfuse_cls: Any = None
        self._handler_cls: Any = None

    def _load_optional_tracing(self) -> bool:
        if self._langfuse_cls is not None and self._handler_cls is not None:
            return True
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler
        except Exception:
            return False
        self._langfuse_cls = Langfuse
        self._handler_cls = CallbackHandler
        return True

    def generate_session_id(self) -> str:
        team = re.sub(r"[^A-Za-z0-9_-]+", "-", self.settings.team_name.strip()) or "team"
        self._session_id = f"{team}-{ulid.new().str}"
        return self._session_id

    def is_enabled(self) -> bool:
        return bool(
            self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
            and self._load_optional_tracing()
        )

    def get_langfuse_client(self):
        if not self.is_enabled():
            return None
        if self._langfuse_client is None:
            try:
                self._langfuse_client = self._langfuse_cls(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                )
            except Exception:
                return None
        return self._langfuse_client

    def get_callback_handler(self):
        if not self.is_enabled():
            return None
        if self._callback_handler is None:
            try:
                client = self.get_langfuse_client()
                if client is None:
                    return None
                self._callback_handler = self._handler_cls()
            except Exception:
                return None
        return self._callback_handler

    def get_langchain_config(self, dataset_name: str, task_name: str) -> dict:
        if not self._session_id:
            self.generate_session_id()
        config: dict = {
            "metadata": {
                "langfuse_session_id": self._session_id,
                "dataset_name": dataset_name,
                "task_name": task_name,
            }
        }
        handler = self.get_callback_handler()
        if handler is not None:
            config["callbacks"] = [handler]
        return config

    def flush(self) -> None:
        client = self.get_langfuse_client()
        if client is not None:
            try:
                client.flush()
            except Exception:
                return None
