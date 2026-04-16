from __future__ import annotations

import re

from src.config import Settings
from src.tracing import TracingManager


def test_session_id_format_teamname_ulid():
    settings = Settings.from_env_and_file()
    settings.team_name = "my team"
    manager = TracingManager(settings)
    session_id = manager.generate_session_id()

    assert session_id.startswith("my-team-")
    ulid_part = session_id.rsplit("-", 1)[-1]
    assert re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", ulid_part) is not None


def test_langchain_config_contains_langfuse_session_metadata():
    settings = Settings.from_env_and_file()
    manager = TracingManager(settings)
    cfg = manager.get_langchain_config(dataset_name="the_truman_show", task_name="communication")

    assert "metadata" in cfg
    assert "langfuse_session_id" in cfg["metadata"]
    assert cfg["metadata"]["dataset_name"] == "the_truman_show"
    assert cfg["metadata"]["task_name"] == "communication"
