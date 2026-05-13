from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, cast
import uuid

from verifiers.types import State as VFState

from .utils.timing_utils import timing_record

if TYPE_CHECKING:
    from .runtime import Runtime
    from .utils.endpoint_utils import EndpointApi
    from verifiers.types import ClientType


_MISSING = object()

BorrowTarget = Literal["model", "sandbox"]
ToolTarget = str | Iterable[str]
TranscriptMode = Literal["private", "append"]


class ForTask(Protocol):
    def __call__(
        self,
        task: Mapping[str, object],
        *,
        borrow: BorrowTarget | Iterable[BorrowTarget] = (),
        tools: ToolTarget = (),
        transcript: TranscriptMode = "private",
    ) -> State: ...


class _StateForTask:
    @overload
    def __get__(self, instance: None, owner: type[State]) -> ForTask: ...

    @overload
    def __get__(self, instance: State, owner: type[State]) -> ForTask: ...

    def __get__(
        self, instance: State | None, owner: type[State]
    ) -> Callable[..., State]:
        def create(
            task: Mapping[str, object],
            *,
            borrow: BorrowTarget | Iterable[BorrowTarget] = (),
            tools: ToolTarget = (),
            transcript: TranscriptMode = "private",
        ) -> State:
            state = _state_for_task(owner, task)
            if instance is not None:
                _borrow_from_state(state, instance, borrow, tools, transcript)
            elif borrow or tools:
                raise ValueError("State.for_task borrow/tools requires a source state.")
            elif transcript != "private":
                raise ValueError(
                    "State.for_task transcript='append' requires a source state."
                )
            return state

        return create


class State(VFState):
    for_task = _StateForTask()

    INTERNAL_KEYS = {"is_completed", "stop_condition", "is_truncated", "error"}
    RUNTIME_HANDLE_KEYS = {"runtime_id", "client_key"}
    ENDPOINT_HANDLE_KEYS = {
        "endpoint_rollout_key",
        "endpoint_root_url",
        "endpoint_base_url",
    }

    def __init__(self, *args: object, **kwargs: Any):
        values = dict(*args, **kwargs)
        protected = sorted(set(values) & self.INTERNAL_KEYS)
        if protected:
            raise RuntimeError(
                f"State constructor cannot set framework-managed keys: {protected}."
            )
        super().__init__(values)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.INTERNAL_KEYS:
            raise RuntimeError(internal_key_error(key))
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        if key in self.INTERNAL_KEYS:
            raise RuntimeError(internal_key_error(key))
        super().__delitem__(key)

    def update(self, *args: object, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[str(key)] = value

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if key in self.INTERNAL_KEYS:
            raise RuntimeError(internal_key_error(key))
        if default is _MISSING:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        raise RuntimeError("State.popitem() cannot preserve framework-managed fields.")

    def clear(self) -> None:
        raise RuntimeError("State.clear() cannot preserve framework-managed fields.")

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self.INTERNAL_KEYS:
            raise RuntimeError(internal_key_error(key))
        return super().setdefault(key, default)

    def __ior__(self, other: object) -> State:
        self.update(other)
        return self

    def _set_internal(self, key: str, value: Any) -> None:
        if key not in self.INTERNAL_KEYS:
            raise KeyError(f"{key!r} is not a framework-managed state key.")
        super().__setitem__(key, value)

    def _set_completed(self, value: bool = True) -> None:
        self._set_internal("is_completed", value)

    def _set_error(self, value: Any) -> None:
        self._set_internal("error", value)

    def _set_stop_condition(
        self, value: str | None, *, overwrite: bool = False
    ) -> None:
        if overwrite or self.get("stop_condition") is None:
            self._set_internal("stop_condition", value)

    def _set_truncated(self, value: bool = True, *, overwrite: bool = False) -> None:
        current = bool(self.get("is_truncated", False))
        self._set_internal(
            "is_truncated", bool(value) if overwrite else current or bool(value)
        )

    def stop(self, condition: str = "state_done") -> None:
        if not isinstance(condition, str) or not condition:
            raise TypeError("State.stop condition must be a non-empty string.")
        super().__setitem__("done", True)
        self._set_completed(True)
        self._set_stop_condition(condition, overwrite=True)

    def runtime_state(self) -> dict[str, object]:
        raw_runtime = self.setdefault("runtime", {})
        if not isinstance(raw_runtime, dict):
            raise TypeError("state.runtime must be a mapping.")
        return cast(dict[str, object], raw_runtime)

    def _runtime(self) -> Runtime:
        from .utils.runtime_registry import load_runtime_from_state

        return load_runtime_from_state(self)

    def get_model(self) -> str:
        runtime = self.get("runtime", {})
        if isinstance(runtime, Mapping):
            model = runtime.get("model")
            if isinstance(model, str) and model:
                return model
            resolved = runtime.get("resolved")
            if isinstance(resolved, Mapping):
                handle = resolved.get("model")
                if isinstance(handle, Mapping):
                    model = handle.get("model")
                    if isinstance(model, str) and model:
                        return model
        try:
            return self._runtime().model(self)
        except RuntimeError as exc:
            raise RuntimeError("State has no resolved model.") from exc

    def get_max_turns(self, default: int) -> int:
        runtime = self.get("runtime", {})
        if isinstance(runtime, Mapping) and "max_turns" in runtime:
            value = runtime["max_turns"]
            if value is None:
                return default
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("state.runtime.max_turns must be an integer.")
            return value
        return default

    def get_client(
        self,
        api: EndpointApi | ClientType = "chat_completions",
        *,
        sync: bool = False,
    ) -> object:
        from .utils.endpoint_utils import client_from_state

        return client_from_state(self, api, sync=sync)

    def get_endpoint_config(
        self,
        api: EndpointApi | ClientType = "chat_completions",
    ) -> dict[str, str]:
        from .utils.endpoint_utils import endpoint_config_from_state

        return endpoint_config_from_state(self, api)

    def get_tools(self) -> dict[str, Callable[..., Awaitable[object]]]:
        from .utils.tool_utils import load_tools_from_state

        return load_tools_from_state(self)

    def _runtime_handles(self) -> dict[str, object]:
        runtime = self.runtime_state()
        handles = runtime.setdefault("resolved", {})
        if not isinstance(handles, dict):
            raise TypeError("state.runtime.resolved must be a mapping.")
        return cast(dict[str, object], handles)

    def _runtime_handle(self, name: str) -> dict[str, object]:
        runtime = self.runtime_state()
        handles = runtime.get("resolved")
        if handles is not None:
            if not isinstance(handles, Mapping):
                raise TypeError("state.runtime.resolved must be a mapping.")
            handles = cast(Mapping[str, object], handles)
            existing = handles.get(name)
            if existing is not None:
                if not isinstance(existing, Mapping):
                    raise TypeError(f"state.runtime.resolved.{name} must be a mapping.")
                return dict(cast(Mapping[str, object], existing))

        runtime_id = runtime.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise RuntimeError("State has no live runtime id.")
        if name == "model":
            client_key = runtime.get("client_key")
            if not isinstance(client_key, str) or not client_key:
                raise RuntimeError("State has no resolved model client.")
            handle: dict[str, object] = {
                "runtime_id": runtime_id,
                "client_key": client_key,
            }
            for key in ("model", "client_type", "sampling_args"):
                if key in runtime:
                    handle[key] = runtime[key]
            return handle
        if name == "endpoint":
            return {"runtime_id": runtime_id}
        if name == "trajectory":
            runtime_obj = self._runtime()
            runtime_obj.register_trajectory(self)
            trajectory = self.get("trajectory") or []
            if not isinstance(trajectory, list):
                raise TypeError("state.trajectory must be a list.")
            return {
                "runtime_id": runtime_id,
                "trajectory_id": str(self["trajectory_id"]),
                "start": len(trajectory),
            }
        if name == "sandbox":
            sandbox = runtime.get("sandbox")
            if not isinstance(sandbox, Mapping):
                raise RuntimeError("State has no resolved primary sandbox.")
            handle = dict(cast(Mapping[str, object], sandbox))
            handle["runtime_id"] = runtime_id
            return handle
        raise KeyError(f"Unknown runtime handle {name!r}.")

    def _tools_handle(self, names: ToolTarget) -> dict[str, object] | None:
        tool_names = tuple(_tool_names(names))
        if not tool_names:
            return None
        runtime = self._runtime()
        handle_id = runtime.register_tool_handle(self, tool_names)
        return {
            "runtime_id": runtime.runtime_id,
            "handle_id": handle_id,
            "names": list(tool_names),
        }

    def _use_runtime_handle(self, name: str, handle: Mapping[str, object]) -> State:
        self._runtime_handles()[name] = dict(handle)
        return self

    def strip_runtime_handles(self) -> None:
        strip_runtime_handles(self)

    def finalize(self) -> State:
        self.strip_runtime_handles()
        self.assert_serializable()
        return self


__all__ = ["State"]


def internal_key_error(key: str) -> str:
    if key == "is_completed":
        return (
            "state['is_completed'] is framework-managed; use state.stop(...), "
            "state['done'], or @vf.stop."
        )
    if key == "stop_condition":
        return (
            "state['stop_condition'] is framework-managed; use state.stop(...), "
            "state['done'], or @vf.stop."
        )
    if key == "is_truncated":
        return (
            "state['is_truncated'] is framework-managed; raise an overlong-prompt "
            "error or let trajectory sync set it."
        )
    if key == "error":
        return "state['error'] is framework-managed; raise vf.Error instead."
    return f"state[{key!r}] is framework-managed."


def _state_for_task(cls: type[State], task: Mapping[str, object]) -> State:
    state = cls(
        {
            "task": dict(task),
            "runtime": {},
            "trajectory": [],
            "trajectory_id": uuid.uuid4().hex,
            "artifacts": {},
            "metrics": {},
            "reward": 0.0,
            "completion": None,
            "timing": timing_record(),
        }
    )
    state._set_completed(False)
    state._set_truncated(False, overwrite=True)
    state._set_stop_condition(None, overwrite=True)
    state._set_error(None)
    for key in ("prompt", "info", "example_id"):
        if key in task:
            state[key] = deepcopy(task[key])
    return state


def _borrow_from_state(
    state: State,
    source: State,
    borrow: BorrowTarget | Iterable[BorrowTarget],
    tools: ToolTarget,
    transcript: TranscriptMode,
) -> None:
    if transcript not in {"private", "append"}:
        raise ValueError("transcript must be 'private' or 'append'.")
    for name in _borrow_targets(borrow):
        if name not in {"model", "sandbox"}:
            raise KeyError(f"Unknown borrow target {name!r}.")
        state._use_runtime_handle(name, source._runtime_handle(name))
    tools_handle = source._tools_handle(tools)
    if tools_handle is not None:
        state._use_runtime_handle("tools", tools_handle)
    if transcript == "append":
        state._use_runtime_handle("trajectory", source._runtime_handle("trajectory"))


def _borrow_targets(
    borrow: BorrowTarget | Iterable[BorrowTarget],
) -> Iterable[BorrowTarget]:
    if isinstance(borrow, str):
        return (cast(BorrowTarget, borrow),)
    return borrow


def _tool_names(tools: ToolTarget) -> Iterable[str]:
    if isinstance(tools, str):
        return (tools,)
    return tools


def strip_runtime_handles(value: object) -> None:
    if isinstance(value, State) or type(value) is dict:
        mapping = cast(dict[str, object], value)
        for key in State.RUNTIME_HANDLE_KEYS:
            mapping.pop(key, None)
        runtime = mapping.get("runtime")
        if type(runtime) is dict:
            runtime_mapping = cast(dict[str, object], runtime)
            runtime_mapping.pop("resolved", None)
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
