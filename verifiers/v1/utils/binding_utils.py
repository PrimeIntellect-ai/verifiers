import inspect
from collections.abc import Mapping, Set
from typing import Literal, TypeAlias, cast
from ..types import ConfigMap, Handler, Objects


BindingRoot: TypeAlias = Literal[
    "task", "state", "tasks", "states", "runtime", "objects", "tools"
]
CallableBindingSource: TypeAlias = Handler | ConfigMap
BindingSource: TypeAlias = str | CallableBindingSource
BindingMap: TypeAlias = dict[str, BindingSource]
Bindings: TypeAlias = dict[str, BindingSource]

VALID_BINDING_ROOTS: frozenset[str] = frozenset(
    {"task", "state", "tasks", "states", "runtime", "objects", "tools"}
)
ROLLOUT_FRAMEWORK_ARGS: frozenset[str] = frozenset(
    {
        "answer",
        "completion",
        "error",
        "example_id",
        "info",
        "metrics",
        "prompt",
        "question",
        "reward",
        "runtime",
        "state",
        "task",
        "task_id",
        "timing",
        "trajectory",
    }
)
GROUP_FRAMEWORK_ARGS: frozenset[str] = frozenset({"states", "tasks"})


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
            normalized_source = cast(ConfigMap, source)
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
        validate_callable_source(cast(ConfigMap, source), context)


def validate_callable_source(source: ConfigMap, context: str) -> None:
    if "fn" not in source:
        raise TypeError(f"{context} mapping sources must use an 'fn' key.")
    unknown = set(source) - {"fn"}
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}.")


def function_name(fn: Handler) -> str:
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
        "Binding string sources must start with task, state, tasks, states, "
        f"runtime, objects, or tools; got {source!r}."
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
    fn: Handler | object,
    arg_name: str,
    context: str,
    protected_args: Set[str] = frozenset(),
) -> None:
    if arg_name in protected_args:
        return
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


def same_callable(left: Handler, right: Handler) -> bool:
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
            current = cast(ConfigMap, current)[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
