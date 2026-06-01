import inspect
from typing import cast

from verifiers.types import Tool
from verifiers.v1.state import State
from verifiers.v1.toolset import Toolset, tool_name
from ..types import ConfigData, RuntimeCallable


def load_tools_from_state(state: State) -> dict[str, RuntimeCallable]:
    runtime = state._runtime()
    task = runtime.task_for_state(state)
    return runtime.tool_calls(task, state)


def tool_error_content(error: Exception) -> str:
    return str(error)


def tool_visible(toolset: Toolset, name: str) -> bool:
    if toolset.show is not None and name not in toolset.show:
        return False
    if toolset.hide is not None and name in toolset.hide:
        return False
    return True


def toolset_object_scope(toolset: Toolset) -> str:
    if toolset.scope is not None:
        return toolset.scope
    return "rollout" if toolset.write else "global"


def schema_callable(tool: object, signature: inspect.Signature) -> RuntimeCallable:
    def call_for_schema() -> None:
        return None

    call_for_schema.__name__ = tool_name(tool)
    call_for_schema.__doc__ = inspect.getdoc(tool)
    setattr(call_for_schema, "__signature__", signature)
    return call_for_schema


def tool_schema(tool: Tool, hidden_args: set[str]) -> Tool:
    parameters = dict(tool.parameters)
    properties = dict(cast(ConfigData, parameters.get("properties") or {}))
    for arg_name in hidden_args:
        properties.pop(arg_name, None)
    parameters["properties"] = properties
    required = parameters.get("required")
    if isinstance(required, list):
        parameters["required"] = [arg for arg in required if arg not in hidden_args]
    return Tool(
        name=tool.name,
        description=tool.description,
        parameters=parameters,
        strict=tool.strict,
    )
