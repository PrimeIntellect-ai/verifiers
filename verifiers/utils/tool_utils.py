import inspect
from copy import copy
from typing import Any
from typing import get_type_hints

from openai import pydantic_function_tool
from pydantic import Field, create_model
from pydantic.fields import FieldInfo

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


def convert_func_to_tool_def(func: Any) -> Tool:
    """Convert *func* to a provider-agnostic vf.Tool definition."""
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n\n", 1)[0]
    param_descriptions: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:"}:
            in_args = True
            continue
        if not in_args:
            continue
        if not line.startswith((" ", "\t")):
            break
        name, separator, param_description = stripped.partition(":")
        if separator:
            param_descriptions[name.split(" ", 1)[0]] = param_description.strip()

    fields: dict[str, Any] = {}
    type_hints = get_type_hints(func, include_extras=True)
    for name, param in inspect.signature(func).parameters.items():
        annotation = type_hints.get(name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        param_description = param_descriptions.get(name)
        if isinstance(default, FieldInfo) and param_description:
            default = copy(default)
            default.description = param_description
        elif param_description:
            default = Field(default=default, description=param_description)
        fields[name] = (annotation, default)

    model = create_model(f"{func.__name__}_args", **fields)
    function = pydantic_function_tool(
        model, name=func.__name__, description=description
    )["function"]
    return Tool(
        name=function["name"],
        description=function.get("description", ""),
        parameters=function["parameters"],
        strict=function["strict"],
    )
