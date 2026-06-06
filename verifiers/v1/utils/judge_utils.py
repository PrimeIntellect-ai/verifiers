import json
from typing import cast


from ..types import JsonData


def parse_judge_json(text: str) -> JsonData:
    value = parsed_json_object(text)
    if value is not None:
        return value
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = parsed_json_object(text[start : end + 1])
        if value is not None:
            return value
    return {"score": 0.0, "reason": "judge did not return JSON", "raw": text}


def parsed_json_object(text: str) -> JsonData | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return cast(JsonData, value)
    return None


def clamp_float(value: object) -> float:
    if not isinstance(value, int | float | str) or isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def truncate_command_record(record: object) -> object:
    if not isinstance(record, dict):
        return record
    record = cast(JsonData, record)
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
