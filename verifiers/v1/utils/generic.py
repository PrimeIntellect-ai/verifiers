"""Generic type resolution and config plumbing shared by v1 plugin classes."""

from typing import TypeVar, get_args, get_origin

from pydantic import ValidationError

T = TypeVar("T")


def prefix_validation_error(e: ValidationError, prefix: tuple) -> ValidationError:
    """`e` with `prefix` prepended to every error's loc. A sub-model validated
    inside a `mode="before"` validator surfaces its errors at the validator's own
    loc, so without re-raising prefixed the CLI renders a flag path missing the
    segments the user actually typed."""
    return ValidationError.from_exception_data(
        e.title,
        [
            {**err, "loc": prefix + tuple(err["loc"])}
            for err in e.errors(include_url=False)
        ],
    )


def deep_merge(base: dict, override: dict) -> dict:
    """`override` onto `base`, recursing into dicts, so a partial nested override
    keeps the untouched keys of the declared default. An override that switches a
    subtree's discriminator (`id`/`type`) replaces the subtree wholesale — the old
    plugin's fields must not leak into the new type's validation."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            switched = any(
                k in value and k in merged[key] and value[k] != merged[key][k]
                for k in ("id", "type")
            )
            merged[key] = value if switched else deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
