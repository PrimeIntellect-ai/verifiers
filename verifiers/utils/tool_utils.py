from typing import Any

from agents.function_schema import function_schema
from openai.types.chat import ChatCompletionFunctionToolParam

VALID_TOOL_CONTENT_PART_TYPES = frozenset({"text", "image_url"})


def is_valid_tool_content_parts(value: Any) -> bool:
    """Check if value is a valid list of tool content parts.

    Valid content parts have a "type" field with value "text" or "image_url".
    """
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if item.get("type") not in VALID_TOOL_CONTENT_PART_TYPES:
            return False
    return True


def convert_func_to_oai_tool(func: Any) -> ChatCompletionFunctionToolParam:
    """Convert *func* to an OpenAI function-calling tool schema.
    The returned mapping matches the structure expected in the `tools` list
    of the OpenAI ChatCompletion API.
    """
    function_schema_obj = function_schema(func)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": function_schema_obj.description or "",
            "parameters": function_schema_obj.params_json_schema,
        },
    }
