from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from verifiers.types import State as VFState

if TYPE_CHECKING:
    from .runtime import Runtime


class State(VFState):
    RUNTIME_HANDLE_KEYS = {"runtime_id", "client_key"}
    ENDPOINT_HANDLE_KEYS = {
        "endpoint_request_key",
        "endpoint_root_url",
        "endpoint_base_url",
    }

    @classmethod
    def for_task(cls, task: Mapping[str, Any]) -> State:
        state = cast(State, super().for_task(task))
        state.pop("answer", None)
        state["runtime"] = {}
        return state

    def runtime_state(self) -> dict[str, Any]:
        raw_runtime = self.setdefault("runtime", {})
        if not isinstance(raw_runtime, dict):
            raise TypeError("state.runtime must be a mapping.")
        return raw_runtime

    def runtime(self) -> Runtime:
        from .runtime import load_runtime_from_state

        return load_runtime_from_state(self)

    def tools(self) -> dict[str, Callable[..., Awaitable[object]]]:
        from .utils.tool_utils import load_tools_from_state

        return load_tools_from_state(self)

    def strip_runtime_handles(self) -> None:
        strip_runtime_handles(self)

    def finalize(self) -> State:
        self.strip_runtime_handles()
        self.assert_serializable()
        return self


__all__ = ["State"]


def strip_runtime_handles(value: object) -> None:
    if isinstance(value, State) or type(value) is dict:
        mapping = cast(dict[str, object], value)
        for key in State.RUNTIME_HANDLE_KEYS:
            mapping.pop(key, None)
        runtime = mapping.get("runtime")
        if type(runtime) is dict:
            runtime_mapping = cast(dict[str, object], runtime)
            for key in State.RUNTIME_HANDLE_KEYS:
                runtime_mapping.pop(key, None)
            sandbox = runtime_mapping.get("sandbox")
            if type(sandbox) is dict:
                cast(dict[str, object], sandbox).pop("lease_key", None)
        for key in State.ENDPOINT_HANDLE_KEYS:
            mapping.pop(key, None)
        for item in list(mapping.values()):
            strip_runtime_handles(item)
        return
    if isinstance(value, list):
        for item in value:
            strip_runtime_handles(item)
