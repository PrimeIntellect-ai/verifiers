"""Generic helpers shared across v1 modules."""

from typing import TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

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


def merge_defaults(defaults: BaseModel, raw: dict | None) -> dict:
    """A raw config block over `defaults`, e.g. a group's shared knobs under each of its
    members. Only the fields set in `defaults` apply, and `raw`'s values win. A block
    whose `id`/`type` differs from the default's is `raw`'s alone, so one plugin's knobs
    never leak into another's."""
    keys = {
        k: getattr(defaults, k)
        for k in ("id", "type")
        if k in type(defaults).model_fields
    }
    block = {**keys, **_dump_set(defaults)}
    return deep_merge({"block": block}, {"block": raw or {}})["block"]


def _dump_set(config: BaseModel) -> dict:
    """`config`'s set fields, including those set on a sub-config after construction
    (which leaves the parent's field unmarked). Each dumped sub-config keeps its
    `id`/`type` even at the default: `deep_merge` detects a plugin switch only when
    both sides name it."""
    dump = {}
    for name in type(config).model_fields:
        value = getattr(config, name)
        if isinstance(value, BaseModel):
            nested = _dump_set(value)
            if nested or name in config.model_fields_set:
                keys = [k for k in ("id", "type") if k in type(value).model_fields]
                dump[name] = {**{k: getattr(value, k) for k in keys}, **nested}
        elif name in config.model_fields_set:
            dump[name] = config.model_dump(include={name})[name]
    return dump


def concrete_type(
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
