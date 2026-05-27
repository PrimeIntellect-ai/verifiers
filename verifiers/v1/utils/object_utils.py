import inspect
from collections.abc import Awaitable, Callable
from typing import cast

from verifiers.utils.async_utils import maybe_call_with_named_args


async def close_object(obj: object) -> None:
    for name in ("aclose", "close", "delete", "teardown"):
        fn = getattr(obj, name, None)
        if callable(fn):
            await maybe_call_with_named_args(fn)
            return


async def resolve_object_factory(
    spec: object, context: str, kwargs: dict[str, object] | None = None
) -> object:
    if not callable(spec):
        raise TypeError(f"{context} must be an import ref or factory function.")
    if not (inspect.isfunction(spec) or inspect.isclass(spec)):
        raise TypeError(f"{context} must be a factory function or class.")
    validate_object_factory(spec, context, kwargs or {})
    value = cast(Callable[..., object | Awaitable[object]], spec)(**(kwargs or {}))
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


def validate_object_loader_spec(spec: object, context: str) -> None:
    if isinstance(spec, str):
        return
    if not callable(spec):
        raise TypeError(f"{context} must be an import ref or factory function.")
    if not (inspect.isfunction(spec) or inspect.isclass(spec)):
        raise TypeError(f"{context} must be a factory function or class.")
    validate_object_factory_spec(spec, context)


def validate_object_factory_spec(spec: object, context: str) -> None:
    if not inspect.isclass(spec):
        name = getattr(spec, "__name__", "")
        if name == "<lambda>":
            raise TypeError(f"{context} must be a named factory function.")
    try:
        inspect.signature(cast(Callable[..., object], spec))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} factory signature cannot be inspected.") from exc


def validate_object_factory(
    spec: object, context: str, kwargs: dict[str, object]
) -> None:
    validate_object_factory_spec(spec, context)
    signature = inspect.signature(cast(Callable[..., object], spec))
    try:
        signature.bind(**kwargs)
    except TypeError as exc:
        raise TypeError(f"{context} has unbound factory arguments.") from exc
