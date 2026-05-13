from __future__ import annotations

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
        return spec
    try:
        signature = inspect.signature(spec)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} factory signature cannot be inspected.") from exc
    if signature.parameters:
        raise TypeError(f"{context} factory must accept no arguments.")
    value = cast(Callable[[], object], spec)()
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value
