import json
from typing import cast
from ..types import ConfigData


def json_args(value: str) -> ConfigData:
    raw = value or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw if len(raw) <= 500 else f"{raw[:500]}..."
        raise ValueError(
            "Invalid JSON tool-call arguments: "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}; "
            f"raw={snippet!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return cast(ConfigData, parsed)
