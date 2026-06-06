import json
from typing import cast

from pydantic import BaseModel

from ..types import JsonData


def json_args(value: str) -> JsonData:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Tool call arguments must decode to a JSON object.")
    return cast(JsonData, parsed)


def jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="json", exclude_none=True))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value
