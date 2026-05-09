from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast


def parse_judge_json(text: str) -> dict[str, object]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return cast(dict[str, object], value)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return cast(dict[str, object], value)
        except json.JSONDecodeError:
            pass
    return {"score": 0.0, "reason": "judge did not return JSON", "raw": text}


def clamp_float(value: object) -> float:
    if not isinstance(value, int | float | str) or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def truncate_command_record(record: object) -> object:
    if not isinstance(record, Mapping):
        return record
    record = cast(Mapping[str, object], record)
    return {
        **dict(record),
        "command": truncate_text(str(record.get("command") or ""), limit=2_000),
        "stdout": truncate_text(str(record.get("stdout") or "")),
        "stderr": truncate_text(str(record.get("stderr") or "")),
    }


def truncate_text(text: str, limit: int = 6_000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


def completion_text(completion: object) -> str:
    if isinstance(completion, list):
        for message in reversed(completion):
            if not isinstance(message, Mapping):
                continue
            message = cast(Mapping[str, object], message)
            if message.get("role") == "assistant":
                return str(message.get("content") or "")
    return str(completion or "")
