from __future__ import annotations

import json
from typing import cast


def json_args(value: str) -> dict[str, object]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return cast(dict[str, object], parsed)
