import inspect
from collections.abc import Set
from typing import Literal, TypeAlias, cast

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from ..config import Config, validate_serializable_value
from ..types import ConfigData, Handler, ObjectFactory, Objects
from .config_utils import resolve_config_object
from .object_utils import validate_object_factory_spec, validate_object_loader_spec


BindingRoot: TypeAlias = Literal[
    "task",
    "state",
    "tasks",
    "states",
    "runtime",
    "objects",
    "tools",
    "taskset",
    "harness",
]
CallableBindingSource: TypeAlias = Handler | ConfigData
BindingSource: TypeAlias = str | CallableBindingSource
BindingSources: TypeAlias = dict[str, BindingSource]
ObjectRefs: TypeAlias = dict[str, str]


class BindingsConfig(Config):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def validate_mapping_input(cls, value: object) -> object:
        if isinstance(value, BindingsConfig):
            return value
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("BindingsConfig must be a mapping.")
        for key, source in value.items():
            if not isinstance(key, str):
                raise TypeError("BindingsConfig keys must be strings.")
            validate_serializable_value(source, f"bindings.{key}")
            validate_binding_source(source, f"bindings source for {key!r}")
        return value

    @model_validator(mode="after")
    def validate_config_entries(self) -> Self:
        for key, source in self.raw_entries().items():
            validate_serializable_value(source, f"bindings.{key}")
            validate_binding_source(source, f"bindings source for {key!r}")
        return self

    def raw_entries(self) -> BindingSources:
        return cast(BindingSources, dict(self.model_extra or {}))

    def entries(
        self,
        field: str = "bindings",
        *,
        allow_objects: bool = True,
        validate_sources: bool = True,
        key_style: Literal["callable", "arg"] = "callable",
    ) -> BindingSources:
        result: BindingSources = {}
        for raw_key, source in self.raw_entries().items():
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
            result[raw_key] = source
        return result


def binding_sources(
    value: BindingSources | BindingsConfig | None,
    field: str = "bindings",
) -> BindingSources:
    if value is None:
        return {}
    if isinstance(value, BindingsConfig):
        return value.entries(field)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a mapping.")
    result: BindingSources = {}
    for raw_key, source in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{field} keys must be strings.")
        binding_key_parts(raw_key)
        validate_binding_source(source, f"{field} source for {raw_key!r}")
        result[raw_key] = source
    return result


class ObjectsConfig(Config):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def validate_mapping_input(cls, value: object) -> object:
        if isinstance(value, ObjectsConfig):
            return value
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("ObjectsConfig must be a mapping.")
        for key, source in value.items():
            if not isinstance(key, str):
                raise TypeError("ObjectsConfig keys must be strings.")
            if not isinstance(source, str):
                raise TypeError(f"objects entry {key!r} must be an import ref string.")
            validate_object_loader_spec(source, f"objects entry {key!r}")
        return value

    def refs(self) -> ObjectRefs:
        return cast(ObjectRefs, dict(self.model_extra or {}))

    def objects(self, field: str = "objects") -> Objects:
        resolved: Objects = {}
        for name, source in self.refs().items():
            factory = resolve_config_object(source)
            validate_object_factory_spec(factory, f"{field}.{name}")
            resolved[name] = cast(ObjectFactory, factory)
        return resolved

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        for key, source in self.refs().items():
            if not isinstance(source, str):
                raise TypeError(f"objects entry {key!r} must be an import ref string.")
            validate_object_loader_spec(source, f"objects entry {key!r}")
        return self


VALID_BINDING_ROOTS: frozenset[str] = frozenset(
    {
        "task",
        "state",
        "tasks",
        "states",
        "runtime",
        "objects",
        "tools",
        "taskset",
        "harness",
    }
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


def validate_binding_source(
    source: object, context: str, *, allow_objects: bool = True
) -> None:
    if (
        not isinstance(source, str)
        and not callable(source)
        and not isinstance(source, dict)
    ):
        raise TypeError(f"{context} must be a framework path or callable.")
    root = binding_source_root(source)
    validate_binding_source_root(root, context, allow_objects=allow_objects)
    if root == "objects":
        if not isinstance(source, str):
            raise TypeError(f"{context} must be a string source.")
        binding_object_name(source)
    if root in {"taskset", "harness"}:
        if not isinstance(source, str):
            raise TypeError(f"{context} must be a string source.")
        owner_object_name(source)
    if isinstance(source, dict):
        validate_callable_source(cast(ConfigData, source), context)


def validate_callable_source(source: ConfigData, context: str) -> None:
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


def binding_key_parts(key: str) -> tuple[str, str]:
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
        f"runtime, objects, tools, taskset, or harness; got {source!r}."
    )


def validate_binding_source_root(
    root: BindingRoot | None, context: str, *, allow_objects: bool = True
) -> None:
    if root is None:
        return
    if root == "objects" and not allow_objects:
        raise ValueError(f"{context} cannot use objects.* sources.")


def binding_object_name(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("Object binding source must be a string.")
    root, separator, tail = source.partition(".")
    if root != "objects" or not separator:
        raise ValueError("Object binding source must be 'objects.name'.")
    name, _, _ = tail.partition(".")
    if not name:
        raise ValueError("Object binding source must be 'objects.name'.")
    return name


def owner_object_name(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("Owner object binding source must be a string.")
    root, separator, tail = source.partition(".")
    if root not in {"taskset", "harness"} or not separator:
        raise ValueError("Owner object binding source must be 'owner.objects.name'.")
    objects_root, objects_separator, object_tail = tail.partition(".")
    if objects_root != "objects" or not objects_separator:
        raise ValueError("Owner object binding source must be 'owner.objects.name'.")
    name, _, _ = object_tail.partition(".")
    if not name:
        raise ValueError("Owner object binding source must be 'owner.objects.name'.")
    return name


def validate_bound_arg(
    fn: object,
    arg_name: str,
    context: str,
    protected_args: Set[str] = frozenset(),
    *,
    allow_reserved: bool = False,
) -> None:
    if arg_name in protected_args:
        return
    if not allow_reserved and arg_name in {"task", "state", "runtime"}:
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
        if isinstance(current, dict):
            current = cast(ConfigData, current)[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
