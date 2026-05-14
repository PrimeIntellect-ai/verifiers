from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import SupportsIndex

from verifiers.types import assert_json_serializable


def assert_serializable(value: object) -> None:
    assert_json_serializable(value)


class FrozenDict(dict):
    def __deepcopy__(self, memo: dict[int, object]) -> dict[object, object]:
        return {
            deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()
        }

    def __setitem__(self, key: str, value: object) -> None:
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

    def __ior__(self, value: object) -> "FrozenDict":
        raise TypeError("Frozen task mappings are immutable.")


class FrozenList(list):
    def __deepcopy__(self, memo: dict[int, object]) -> list[object]:
        return [deepcopy(value, memo) for value in self]

    def __setitem__(self, key: object, value: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def __delitem__(self, key: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def append(self, value: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def extend(self, values: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def insert(self, index: SupportsIndex, object: object, /) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def pop(self, index: SupportsIndex = -1, /) -> object:
        raise TypeError("Frozen task lists are immutable.")

    def remove(self, value: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def clear(self) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def __iadd__(self, values: Iterable[object]) -> "FrozenList":
        raise TypeError("Frozen task lists are immutable.")

    def __imul__(self, value: SupportsIndex) -> "FrozenList":
        raise TypeError("Frozen task lists are immutable.")

    def sort(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Frozen task lists are immutable.")

    def reverse(self) -> None:
        raise TypeError("Frozen task lists are immutable.")


def freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return FrozenDict({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze_value(item) for item in value)
    return value
