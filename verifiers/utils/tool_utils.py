import inspect
from typing import Any, Callable

from agents.function_schema import function_schema
from openai.types.chat import ChatCompletionFunctionToolParam


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
            "strict": True,
        },
    }


def build_schema_only_tool(tool: Callable, args_to_skip: list[str]) -> Callable:
    """
    Convert a function to an OpenAI/Pydantic-compatible stub tool, excluding specified parameters.

    Args:
        func: The function to convert
        exclude_params: List of parameter names to exclude from the schema
    """
    if not args_to_skip:
        return tool

    original_signature = inspect.signature(tool)

    missing_args = [
        name for name in args_to_skip if name not in original_signature.parameters
    ]
    assert not missing_args, (
        f"{getattr(tool, '__name__')} does not define {missing_args}."
    )

    filtered_parameters = [
        parameter
        for name, parameter in original_signature.parameters.items()
        if name not in args_to_skip
    ]
    schema_signature = original_signature.replace(parameters=filtered_parameters)

    tool_annotations = dict(getattr(tool, "__annotations__", {}))
    for arg in args_to_skip:
        tool_annotations.pop(arg)

    if inspect.iscoroutinefunction(tool):

        async def schema_stub(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "Schema-only stub created for tool registration; this callable should not be invoked."
            )
    else:

        def schema_stub(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "Schema-only stub created for tool registration; this callable should not be invoked."
            )

    schema_stub.__name__ = getattr(tool, "__name__", tool.__class__.__name__)
    schema_stub.__qualname__ = getattr(tool, "__qualname__", schema_stub.__name__)
    schema_stub.__module__ = getattr(tool, "__module__", schema_stub.__module__)
    schema_stub.__doc__ = getattr(tool, "__doc__", schema_stub.__doc__)
    schema_stub.__annotations__ = tool_annotations
    schema_stub.__signature__ = schema_signature  # type: ignore[attr-defined]

    return schema_stub
