from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from verifiers.v1.runtime import Runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task


def load_tools_from_state(
    state: State,
    runtime: Runtime,
) -> dict[str, Callable[..., Awaitable[object]]]:
    task = Task(cast(Mapping[str, Any], state["task"])).freeze()
    return runtime.tool_calls(task, state)
