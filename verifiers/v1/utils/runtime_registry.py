from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..runtime import Runtime

_RUNTIME_REGISTRY: weakref.WeakValueDictionary[str, object] = (
    weakref.WeakValueDictionary()
)


def register_runtime(runtime_id: str, runtime: object) -> None:
    _RUNTIME_REGISTRY[runtime_id] = runtime


def unregister_runtime(runtime_id: str) -> None:
    _RUNTIME_REGISTRY.pop(runtime_id, None)


def load_runtime(runtime_id: str) -> Runtime:
    runtime = _RUNTIME_REGISTRY.get(runtime_id)
    if runtime is None:
        raise RuntimeError(f"No live v1 runtime registered for id {runtime_id!r}.")
    return cast("Runtime", runtime)


def load_runtime_from_state(state: Mapping[str, object]) -> Runtime:
    runtime_state = state.get("runtime")
    if not isinstance(runtime_state, Mapping):
        raise RuntimeError("State has no runtime metadata.")
    runtime_state = cast(Mapping[str, object], runtime_state)
    runtime_id = runtime_state.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        raise RuntimeError("State has no live runtime id.")
    return load_runtime(runtime_id)
