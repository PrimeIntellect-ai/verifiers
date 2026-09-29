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
    """Fill in a raw config block with the values that `defaults` sets, e.g. a
    group's shared knobs under each member's own block.

    - Only fields that were set on `defaults` are filled in; its plain defaults are not.
    - Nested blocks are filled in key by key; a value already in `raw` is kept.
    - If `raw` names a different `id`/`type` for a block (at any depth), that block is
      taken from `raw` alone, so one plugin's fields never leak into another's.

    For example, with `defaults` set to `algo = {type = "grpo", kl = 0.1}`:
    `{}` becomes `{type = "grpo", kl = 0.1}`, `{kl = 0.5}` becomes
    `{type = "grpo", kl = 0.5}`, and `{type = "max_rl"}` stays `{type = "max_rl"}`.
    """
    # `deep_merge` detects an `id`/`type` switch only on nested blocks, so wrap the
    # block once to detect a switch of the block itself too.
    identity = {
        k: getattr(defaults, k)
        for k in ("id", "type")
        if k in type(defaults).model_fields
    }
    block = {**identity, **_dump_set(defaults)}
    return deep_merge({"block": block}, {"block": raw or {}})["block"]


def _dump_set(config: BaseModel) -> dict:
    """The fields set on `config`, as a nested dict. A sub-config the parent did not
    set still counts when its own fields changed after construction (pydantic then
    leaves the parent's field unmarked), but not when it only matches the field's
    default, which may itself be built with arguments. A dumped sub-config always
    includes its `id`/`type`, because `deep_merge` detects a switch only when both
    sides name it."""
    dump = {}
    for name, field in type(config).model_fields.items():
        value = getattr(config, name)
        if isinstance(value, BaseModel):
            nested = _dump_set(value)
            if name not in config.model_fields_set:
                default = field.get_default(call_default_factory=True)
                if isinstance(default, BaseModel) and nested == _dump_set(default):
                    continue
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
