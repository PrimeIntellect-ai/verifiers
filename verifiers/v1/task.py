from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from verifiers.types import assert_json_serializable


class Task(dict):
    def __init__(self, row: Mapping[str, Any] | None = None):
        super().__init__(deepcopy(dict(row or {})))
        self._frozen = False

    def freeze(self) -> Task:
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

    def clear(self) -> None:
        self._raise_if_frozen()
        super().clear()

    def _raise_if_frozen(self) -> None:
        if self._frozen:
            raise TypeError("Task is immutable after freeze.")


def assert_serializable(value: object) -> None:
    assert_json_serializable(value)
