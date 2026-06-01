import inspect
from collections.abc import Awaitable
from typing import cast

from verifiers.utils.async_utils import maybe_call_with_named_args

from ..types import ObjectFactory, RuntimeObject


async def close_object(obj: RuntimeObject) -> None:
    for name in ("aclose", "close", "delete", "teardown"):
        fn = getattr(obj, name, None)
        if callable(fn):
            await maybe_call_with_named_args(fn)
            return


async def resolve_object_factory(
    spec: ObjectFactory, context: str, kwargs: dict[str, RuntimeObject] | None = None
) -> RuntimeObject:
    if not callable(spec):
        raise TypeError(f"{context} must be an import ref or factory function.")
    if not (inspect.isfunction(spec) or inspect.isclass(spec)):
        raise TypeError(f"{context} must be a factory function or class.")
    validate_object_factory(spec, context, kwargs or {})
    value = cast(ObjectFactory, spec)(**(kwargs or {}))
    if inspect.isawaitable(value):
        return await cast(Awaitable[RuntimeObject], value)
    return value


def validate_object_loader_spec(spec: RuntimeObject, context: str) -> None:
    if isinstance(spec, str):
        return
    if not callable(spec):
        raise TypeError(f"{context} must be an import ref or factory function.")
    if not (inspect.isfunction(spec) or inspect.isclass(spec)):
        raise TypeError(f"{context} must be a factory function or class.")
    validate_object_factory_spec(spec, context)


def validate_object_factory_spec(spec: RuntimeObject, context: str) -> None:
    if not inspect.isclass(spec):
        name = getattr(spec, "__name__", "")
        if name == "<lambda>":
            raise TypeError(f"{context} must be a named factory function.")
    try:
        inspect.signature(cast(ObjectFactory, spec))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} factory signature cannot be inspected.") from exc


def validate_object_factory(
    spec: RuntimeObject, context: str, kwargs: dict[str, RuntimeObject]
) -> None:
    validate_object_factory_spec(spec, context)
    signature = inspect.signature(cast(ObjectFactory, spec))
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        raise TypeError(f"{context} has unbound factory arguments.") from exc
