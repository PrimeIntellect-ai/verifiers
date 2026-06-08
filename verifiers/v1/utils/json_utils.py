import json

from pydantic import BaseModel

from ..types import JsonData, JsonValue


def json_args(value: str) -> JsonData:
    parsed = json.loads(value or "{}")
    return json_data(parsed, context="Tool call arguments")


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


def json_value(value: object, *, context: str = "Value") -> JsonValue:
    resolved = jsonable(value)
    if resolved is None or isinstance(resolved, str | int | float | bool):
        return resolved
    if isinstance(resolved, list):
        return [json_value(item, context=context) for item in resolved]
    if isinstance(resolved, dict):
        return {
            str(key): json_value(item, context=f"{context}.{key}")
            for key, item in resolved.items()
        }
    raise TypeError(f"{context} must be JSON serializable.")


def json_data(value: object, *, context: str = "Value") -> JsonData:
    resolved = json_value(value, context=context)
    if not isinstance(resolved, dict):
        raise TypeError(f"{context} must be a JSON object.")
    return resolved
