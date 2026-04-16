from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_ascii_lines(path: str, lines: list[str]) -> None:
    ensure_parent_dir(path)
    payload = "\n".join(lines).strip() + "\n"
    payload.encode("ascii")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(payload)


def write_csv(path: str, df: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    df.to_csv(path, index=False)
