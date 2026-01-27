from typing import Any

from agents.function_schema import function_schema

from verifiers.types import Tool


def convert_func_to_tool(func: Any) -> Tool:
    """Convert *func* to an OpenAI function-calling tool schema.
    The returned mapping matches the structure expected in the `tools` list
    of the OpenAI ChatCompletion API.
    """
    function_schema_obj = function_schema(func)
    return Tool(
        name=func.__name__,
        description=function_schema_obj.description or "",
        parameters=function_schema_obj.params_json_schema,
    )
