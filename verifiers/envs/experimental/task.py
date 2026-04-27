from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from verifiers.types import Messages, flatten_task_input


class Task(dict[str, Any]):
    """Serializable task row passed into a harness."""

    __slots__ = ("_frozen",)

    _frozen: bool
    prompt: Messages
    example_id: int
    answer: Any
    info: dict[str, Any]

    def __init__(self, row: Mapping[str, Any] | None = None, **kwargs: Any):
        object.__setattr__(self, "_frozen", False)
        data = dict(row or {})
        data.update(kwargs)
        data = flatten_task_input(data)
        validate_task_serializable(data)
        super().__init__(data)
        self.freeze()

    def freeze(self) -> "Task":
        object.__setattr__(self, "_frozen", True)
        return self

    def _raise_if_frozen(self) -> None:
        if self._frozen:
            raise TypeError("Task rows are immutable after initialization.")

    def __setitem__(self, key: str, value: Any) -> None:
        self._raise_if_frozen()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._raise_if_frozen()
        super().__delitem__(key)

    def clear(self) -> None:
        self._raise_if_frozen()
        super().clear()

    def pop(self, key: str, default: Any = None) -> Any:
        self._raise_if_frozen()
        return super().pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        self._raise_if_frozen()
        return super().popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._raise_if_frozen()
        return super().setdefault(key, default)

    def update(self, *args: object, **kwargs: Any) -> None:
        self._raise_if_frozen()
        super().update(*args, **kwargs)

    def __ior__(self, other: object) -> "Task":
        self._raise_if_frozen()
        super().update(cast(Any, other))
        return self

    @property
    def prompt(self) -> Messages:
        return cast(Messages, self.get("prompt", []))

    @property
    def example_id(self) -> int:
        return int(self.get("example_id", 0))

    @property
    def answer(self) -> Any:
        return self.get("answer", "")

    @property
    def info(self) -> dict[str, Any]:
        value = self.get("info", {})
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise TypeError("task.info must be a mapping.")
        return dict(cast(Mapping[str, Any], value))


def task_to_input(task: Mapping[str, Any]) -> dict[str, Any]:
    input_data = dict(task)
    input_data.setdefault("prompt", [])
    input_data.setdefault("example_id", 0)
    return input_data


def validate_task_serializable(task: Mapping[str, Any]) -> None:
    try:
        json.dumps(task, default=_json_model_dump)
    except TypeError as e:
        raise TypeError("Task rows must be JSON-serializable.") from e


def _json_model_dump(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    raise TypeError
