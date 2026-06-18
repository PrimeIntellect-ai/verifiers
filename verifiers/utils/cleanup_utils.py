from typing import Any

from verifiers.errors import Error
from verifiers.types import State


def append_cleanup_error(
    state: State,
    exc: Exception,
    **metadata: Any,
) -> None:
    cleanup_errors = state.setdefault("cleanup_errors", [])
    if not isinstance(cleanup_errors, list):
        cleanup_errors = []
        state["cleanup_errors"] = cleanup_errors
    record: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    record.update({key: value for key, value in metadata.items() if value is not None})
    cleanup_errors.append(record)


def surface_cleanup_error(
    state: State,
    exc: Exception,
    **metadata: Any,
) -> None:
    append_cleanup_error(state, exc, **metadata)
    if state.get("error") is None:
        error = exc if isinstance(exc, Error) else Error(str(exc))
        set_error = getattr(state, "_set_error", None)
        if callable(set_error):
            set_error(error)
        else:
            state["error"] = error
