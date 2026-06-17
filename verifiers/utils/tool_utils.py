import inspect
from collections.abc import Callable
from typing import Any

from docstring_parser import parse_from_object
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from openai import pydantic_function_tool

from verifiers.types import Tool

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


def convert_func_to_tool_def(func: Callable[..., Any]) -> Tool:
    """Convert *func* to a provider-agnostic vf.Tool definition."""
    name = getattr(func, "__name__")
    doc = parse_from_object(func)
    model = func_metadata(func, structured_output=False).arg_model
    model.model_config["title"] = f"{name}_args"

    fields = {field.alias or name: field for name, field in model.model_fields.items()}
    descriptions = {param.arg_name: param.description for param in doc.params}
    for param in inspect.signature(func).parameters.values():
        field = fields[param.name]
        field.description = field.description or descriptions.get(param.name)
        if param.annotation is inspect.Parameter.empty:
            field.metadata = []
    model.model_rebuild(force=True)

    return Tool.model_validate(
        pydantic_function_tool(
            model,
            name=name,
            description=doc.short_description or "",
        )["function"]
    )
