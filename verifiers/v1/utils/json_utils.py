import json
from typing import cast
from ..types import ConfigData


def json_args(value: str) -> ConfigData:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return cast(ConfigData, parsed)
