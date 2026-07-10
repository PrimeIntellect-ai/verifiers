"""Generic type resolution shared by v1 plugin classes."""

from typing import TypeVar, get_args, get_origin

T = TypeVar("T")


def generic_type(
    cls: type, bound: type[T], *, origin: type | None = None
) -> type[T] | None:
    """Find a concrete bounded type through `cls`'s MRO, most-derived first."""
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            if origin is not None and get_origin(base) is not origin:
                continue
            for arg in get_args(base):
                if isinstance(arg, type) and issubclass(arg, bound):
                    return arg
    return None
