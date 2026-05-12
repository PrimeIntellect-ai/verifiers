from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, SupportsIndex, cast

from verifiers.types import assert_json_serializable

from .config import sandbox_config_mapping
from .utils.prompt_utils import normalize_prompt, normalize_system_prompt


class Task(dict):
    def __init__(self, row: Mapping[str, Any] | None = None):
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
            super().__setitem__(
                "sandbox", sandbox_config_mapping(self["sandbox"], fill_defaults=False)
            )
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

    def __setitem__(self, key: str, value: Any) -> None:
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

    def popitem(self) -> tuple[str, Any]:
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


def assert_serializable(value: object) -> None:
    assert_json_serializable(value)


class FrozenDict(dict):
    def __deepcopy__(self, memo: dict[int, object]) -> dict[object, object]:
        return {
            deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()
        }

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("Frozen task mappings are immutable.")

    def __delitem__(self, key: str) -> None:
        raise TypeError("Frozen task mappings are immutable.")

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Frozen task mappings are immutable.")

    def setdefault(self, key: str, default: object = None) -> object:
        raise TypeError("Frozen task mappings are immutable.")

    def pop(self, key: str, default: object = None) -> object:
        raise TypeError("Frozen task mappings are immutable.")

    def popitem(self) -> tuple[object, object]:
        raise TypeError("Frozen task mappings are immutable.")

    def clear(self) -> None:
        raise TypeError("Frozen task mappings are immutable.")

    def __ior__(self, value: object) -> FrozenDict:
        raise TypeError("Frozen task mappings are immutable.")


class FrozenList(list):
    def __deepcopy__(self, memo: dict[int, object]) -> list[object]:
        return [deepcopy(value, memo) for value in self]

    def __setitem__(self, key: object, value: Any) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def __delitem__(self, key: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def append(self, value: Any) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def extend(self, values: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def insert(self, index: SupportsIndex, object: Any, /) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def pop(self, index: SupportsIndex = -1, /) -> object:
        raise TypeError("Frozen task lists are immutable.")

    def remove(self, value: Any) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def clear(self) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def __iadd__(self, values: Iterable[Any]) -> FrozenList:
        raise TypeError("Frozen task lists are immutable.")

    def __imul__(self, value: SupportsIndex) -> FrozenList:
        raise TypeError("Frozen task lists are immutable.")

    def sort(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def reverse(self) -> None:
        raise TypeError("Frozen task lists are immutable.")


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_value(item) for item in value)
    return value
