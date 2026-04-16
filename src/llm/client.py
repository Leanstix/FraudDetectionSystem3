from __future__ import annotations

from typing import Any

from src.config import Settings
from src.llm.cache import LLMCache
from src.tracing import TracingManager


class LLMClient:
    def __init__(self, settings: Settings, tracing: TracingManager):
        self.settings = settings
        self.tracing = tracing
        self.cache = LLMCache(settings.llm_cache_dir)
        self._model: Any = None
        self._human_message_cls: Any = None
        self._init_error = False

    def _ensure_model(self) -> bool:
        if self._model is not None and self._human_message_cls is not None:
            return True
        if self._init_error or not self.is_enabled():
            return False

        try:
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI
        except Exception:
            self._init_error = True
            return False

        try:
            self._model = ChatOpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model=self.settings.default_model,
                temperature=self.settings.default_temperature,
            )
            self._human_message_cls = HumanMessage
            return True
        except Exception:
            self._init_error = True
            return False

    def is_enabled(self) -> bool:
        return self.settings.is_llm_enabled()

    def invoke(self, prompt: str, dataset_name: str, task_name: str) -> str:
        if not self._ensure_model():
            return ""

        key = self.cache.make_key(self.settings.default_model, prompt)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            config = self.tracing.get_langchain_config(dataset_name=dataset_name, task_name=task_name)
            response = self._model.invoke([self._human_message_cls(content=prompt)], config=config)
            content = response.content if isinstance(response.content, str) else str(response.content)
            self.cache.set(key, content)
            return content
        except Exception:
            return ""
