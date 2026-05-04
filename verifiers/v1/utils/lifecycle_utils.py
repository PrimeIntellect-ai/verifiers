from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Literal, cast

from verifiers.utils.async_utils import maybe_call_with_named_args

LifecycleStage = Literal["rollout", "group"]


def collect_handlers(
    owners: Iterable[object | None],
    attr: str,
    extra: Iterable[Callable[..., object]] = (),
    stage: LifecycleStage | None = None,
) -> list[Callable[..., object]]:
    handlers: list[Callable[..., object]] = []
    for owner in owners:
        if owner is None:
            continue
        for _, method in inspect.getmembers(owner, predicate=callable):
            if getattr(method, attr, False):
                handlers.append(cast(Callable[..., object], method))
    handlers.extend(extra)
    if stage is not None:
        handlers = [
            handler
            for handler in handlers
            if getattr(handler, f"{attr}_stage", "rollout") == stage
        ]
    return sorted(
        unique_handlers(handlers),
        key=lambda handler: (
            -int(getattr(handler, f"{attr}_priority", 0)),
            str(getattr(handler, "__name__", "")),
        ),
    )


def validate_handler_args(
    handlers: Iterable[Callable[..., object]],
    expected: set[str],
    attr: str,
    stage: LifecycleStage,
) -> None:
    expected_text = " and ".join(sorted(expected))
    for handler in handlers:
        names = set(inspect.signature(handler).parameters)
        if names != expected:
            name = str(getattr(handler, "__name__", type(handler).__name__))
            raise ValueError(
                f"{stage} {attr} handler {name!r} must accept exactly {expected_text}."
            )


async def run_handlers(
    handlers: Iterable[Callable[..., object]], **kwargs: object
) -> None:
    for handler in handlers:
        await maybe_call_with_named_args(handler, **kwargs)


def unique_handlers(
    handlers: Iterable[Callable[..., object]],
) -> list[Callable[..., object]]:
    unique: list[Callable[..., object]] = []
    seen: set[tuple[int, int]] = set()
    for handler in handlers:
        key = (
            id(getattr(handler, "__self__", None)),
            id(getattr(handler, "__func__", handler)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(handler)
    return unique
