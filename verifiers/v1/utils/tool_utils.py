from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import cast

from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.toolset import Toolset, tool_name


def load_tools_from_state(
    state: State,
) -> dict[str, Callable[..., Awaitable[object]]]:
    runtime = state._runtime()
    task = Task(cast(Mapping[str, object], state["task"])).freeze()
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


def string_list(value: object, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise TypeError(f"{field} must be a string or list of strings.")
    result = [str(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate names.")
    return result


def schema_callable(
    tool: object, signature: inspect.Signature
) -> Callable[..., object]:
    def call_for_schema(**kwargs: object) -> None:
        _ = kwargs
        return None

    call_for_schema.__name__ = tool_name(tool)
    call_for_schema.__doc__ = getattr(tool, "__doc__", None)
    setattr(call_for_schema, "__signature__", signature)
    return call_for_schema
