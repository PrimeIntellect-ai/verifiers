"""Minimal filesystem cache compatible with QUEST objective evaluators."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class CacheFileSys:
    """Single-task JSON/text cache.

    The generated QUEST scripts pass this object through to the evaluator. The
    evaluator state persistence only needs ``task_dir``; simple get/set helpers
    are provided for compatibility with upstream utility usage.
    """

    def __init__(self, task_dir: str):
        self.task_dir = os.path.abspath(task_dir)
        Path(self.task_dir).mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, suffix: str = ".json") -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return Path(self.task_dir) / f"{digest}{suffix}"

    def get_json(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set_json(self, key: str, value: Any) -> None:
        path = self._path(key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, default=str)

    def get_text(self, key: str) -> str | None:
        path = self._path(key, ".txt")
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def set_text(self, key: str, value: str) -> None:
        self._path(key, ".txt").write_text(value, encoding="utf-8")
