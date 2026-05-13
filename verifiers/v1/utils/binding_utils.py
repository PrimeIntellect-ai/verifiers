from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias, cast

from ..types import Objects

BindingRoot: TypeAlias = Literal["task", "state", "runtime", "objects", "tools"]
CallableBindingSource: TypeAlias = Callable[..., object] | Mapping[str, object]
BindingSource: TypeAlias = str | CallableBindingSource
BindingMap: TypeAlias = Mapping[str, BindingSource]
Bindings: TypeAlias = dict[str, BindingSource]

VALID_BINDING_ROOTS: frozenset[str] = frozenset(
    {"task", "state", "runtime", "objects", "tools"}
)


def normalize_binding_map(
    value: object,
    field: str,
    *,
    allow_objects: bool = True,
    validate_sources: bool = True,
    key_style: Literal["callable", "arg"] = "callable",
) -> Bindings:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    result: Bindings = {}
    for raw_key, source in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{field} keys must be strings.")
        if key_style == "callable":
            binding_key_parts(raw_key)
        elif not raw_key or "." in raw_key:
            raise ValueError(f"{field} keys must be argument names.")
        if validate_sources:
            validate_binding_source(
                source,
                f"{field} source for {raw_key!r}",
                allow_objects=allow_objects,
            )
        if isinstance(source, Mapping):
            normalized_source = cast(Mapping[str, object], source)
        elif isinstance(source, str) or callable(source):
            normalized_source = source
        else:
            raise TypeError(
                f"{field} source for {raw_key!r} must be a framework path or callable."
            )
        result[raw_key] = normalized_source
    return result


def normalize_object_map(value: object, field: str) -> Objects:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    result: Objects = {}
    for raw_key, source in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{field} keys must be strings.")
        if not raw_key:
            raise ValueError(f"{field} keys must be non-empty strings.")
        result[raw_key] = source
    return result


def validate_binding_source(
    source: object, context: str, *, allow_objects: bool = True
) -> None:
    if (
        not isinstance(source, str)
        and not callable(source)
        and not isinstance(source, Mapping)
    ):
        raise TypeError(f"{context} must be a framework path or callable.")
    root = binding_source_root(source)
    validate_binding_source_root(root, context, allow_objects=allow_objects)
    if root == "objects":
        binding_object_name(source)
    if isinstance(source, Mapping):
        validate_callable_source(cast(Mapping[str, object], source), context)


def validate_callable_source(source: Mapping[str, object], context: str) -> None:
    if "fn" not in source:
        raise TypeError(f"{context} mapping sources must use an 'fn' key.")
    unknown = set(source) - {"fn"}
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}.")


def function_name(fn: Callable[..., object]) -> str:
    name = getattr(fn, "__name__", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Callable bindings require a stable __name__.")
    return name


def binding_key_parts(key: object) -> tuple[str, str]:
    if not isinstance(key, str):
        raise TypeError("Binding keys must be strings.")
    target, separator, arg_name = key.partition(".")
    if separator != "." or not target or not arg_name or "." in arg_name:
        raise ValueError(f"Binding key {key!r} must be 'callable.arg'.")
    return target, arg_name


def binding_source_root(source: object) -> BindingRoot | None:
    if not isinstance(source, str):
        return None
    root, _, _ = source.partition(".")
    if root in VALID_BINDING_ROOTS:
        return cast(BindingRoot, root)
    raise ValueError(
        "Binding string sources must start with task, state, runtime, objects, "
        f"or tools; got {source!r}."
    )


def validate_binding_source_root(
    root: BindingRoot | None, context: str, *, allow_objects: bool = True
) -> None:
    if root is None:
        return
    if root == "objects" and not allow_objects:
        raise ValueError(f"{context} cannot use objects.* sources.")


def binding_object_name(source: object) -> str:
    if not isinstance(source, str):
        raise TypeError("Object binding source must be a string.")
    root, separator, tail = source.partition(".")
    if root != "objects" or not separator:
        raise ValueError("Object binding source must be 'objects.name'.")
    name, _, _ = tail.partition(".")
    if not name:
        raise ValueError("Object binding source must be 'objects.name'.")
    return name


def validate_bound_arg(
    fn: Callable[..., object] | object, arg_name: str, context: str
) -> None:
    if arg_name in {"task", "state", "runtime"}:
        raise ValueError(f"{context} cannot bind reserved arg {arg_name!r}.")
    if not callable(fn):
        raise TypeError(f"{context} target is not callable.")
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} target signature cannot be inspected.") from exc
    if arg_name not in signature.parameters:
        name = (
            getattr(fn, "__name__", None)
            or getattr(fn, "name", None)
            or type(fn).__name__
        )
        raise TypeError(
            f"{context} targets {name!r}, but {name!r} does not declare "
            f"arg {arg_name!r}."
        )


def same_callable(left: Callable[..., object], right: Callable[..., object]) -> bool:
    if left is right:
        return True
    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    return left_self is right_self and left_func is not None and left_func is right_func


def read_path(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid empty path segment in {path!r}.")
        if isinstance(current, Mapping):
            current = cast(Mapping[str, object], current)[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
