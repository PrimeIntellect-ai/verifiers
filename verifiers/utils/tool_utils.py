from collections.abc import Mapping
from typing import Any

from agents.function_schema import function_schema

from verifiers.types import Tool

VALID_TOOL_CONTENT_PART_TYPES = frozenset({"text", "image_url"})


def is_valid_tool_content_parts(value: Any) -> bool:
    """Check if value is a valid list of tool content parts.

    Valid content parts have a "type" field with value "text" or "image_url",
    and can be either dict-like objects or pydantic models.
    """
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, Mapping):
            content_type = item.get("type")
        elif hasattr(item, "model_dump"):
            content_type = getattr(item, "type", None)
        else:
            return False
        if content_type not in VALID_TOOL_CONTENT_PART_TYPES:
            return False
    return True


def convert_func_to_tool_def(func: Any) -> Tool:
    """Convert *func* to a provider-agnostic vf.Tool definition."""
    function_schema_obj = function_schema(func)
    return Tool(
        name=func.__name__,
        description=function_schema_obj.description or "",
        parameters=function_schema_obj.params_json_schema,
    )
