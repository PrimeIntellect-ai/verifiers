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


async def resolve_object_factory(spec: object, context: str) -> object:
    if not callable(spec):
        raise TypeError(f"{context} must be an import ref or no-arg loader.")
    if not (inspect.isfunction(spec) or inspect.isclass(spec)):
        raise TypeError(f"{context} must be an importable no-arg loader.")
    validate_no_arg_loader(spec, context)
    value = cast(Callable[[], object | Awaitable[object]], spec)()
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


def validate_object_loader_spec(spec: object, context: str) -> None:
    if isinstance(spec, str):
        return
    if not callable(spec):
        raise TypeError(f"{context} must be an import ref or no-arg loader.")
    if not (inspect.isfunction(spec) or inspect.isclass(spec)):
        raise TypeError(f"{context} must be an importable no-arg loader.")
    validate_no_arg_loader(spec, context)


def validate_no_arg_loader(spec: object, context: str) -> None:
    if not inspect.isclass(spec):
        name = getattr(spec, "__name__", "")
        qualname = getattr(spec, "__qualname__", "")
        module = getattr(spec, "__module__", "")
        if name == "<lambda>" or "<locals>" in qualname or not module:
            raise TypeError(f"{context} must be importable by module path.")
    try:
        signature = inspect.signature(cast(Callable[..., object], spec))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} factory signature cannot be inspected.") from exc
    if signature.parameters:
        raise TypeError(f"{context} factory must accept no arguments.")
