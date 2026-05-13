from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

from .config import sandbox_config_mapping
from .utils.task_freeze_utils import assert_serializable, freeze_value
from .utils.prompt_utils import normalize_prompt, normalize_system_prompt


class Task(dict):
    def __init__(self, row: Mapping[str, object] | None = None):
        super().__init__(deepcopy(dict(row or {})))
        self._frozen = False

    def freeze(self) -> Task:
        if "runtime" in self:
            raise TypeError(
                "task.runtime is not supported; use top-level task fields or state.runtime."
            )
        if "prompt" in self:
            super().__setitem__(
                "prompt", normalize_prompt(self["prompt"], field_name="task.prompt")
            )
        if "system_prompt" in self:
            super().__setitem__(
                "system_prompt",
                normalize_system_prompt(
                    self["system_prompt"], field_name="task.system_prompt"
                ),
            )
        if "tools" in self and not isinstance(self["tools"], Mapping):
            raise TypeError("task.tools must be a mapping with show or hide.")
        if "toolsets" in self and not isinstance(self["toolsets"], Mapping):
            raise TypeError("task.toolsets must be a mapping.")
        if "sandbox" in self and not isinstance(self["sandbox"], Mapping):
            raise TypeError("task.sandbox must be a mapping.")
        if "sandbox" in self:
            super().__setitem__("sandbox", sandbox_config_mapping(self["sandbox"]))
        if "program" in self and not isinstance(self["program"], Mapping):
            raise TypeError("task.program must be a mapping.")
        if "max_turns" in self and (
            not isinstance(self["max_turns"], int)
            or isinstance(self["max_turns"], bool)
        ):
            raise TypeError("task.max_turns must be an integer.")
        for key, value in list(self.items()):
            super().__setitem__(key, freeze_value(value))
        assert_serializable(self)
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __setitem__(self, key: str, value: object) -> None:
        self._raise_if_frozen()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._raise_if_frozen()
        super().__delitem__(key)

    def update(self, *args: object, **kwargs: object) -> None:
        self._raise_if_frozen()
        super().update(*args, **kwargs)

    def setdefault(self, key: str, default: object = None) -> object:
        self._raise_if_frozen()
        return super().setdefault(key, default)

    def pop(self, key: str, default: object = None) -> object:
        self._raise_if_frozen()
        return super().pop(key, default)

    def popitem(self) -> tuple[str, object]:
        self._raise_if_frozen()
        return super().popitem()

    def clear(self) -> None:
        self._raise_if_frozen()
        super().clear()

    def __ior__(self, value: Any, /) -> Task:
        self._raise_if_frozen()
        return cast(Task, dict.__ior__(self, value))

    def _raise_if_frozen(self) -> None:
        if self._frozen:
            raise TypeError("Task is immutable after freeze.")
