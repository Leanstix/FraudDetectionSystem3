from __future__ import annotations


class BaseAgent:
    name: str = "base_agent"

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def validate_inputs(self, *args, **kwargs) -> None:
        return None
