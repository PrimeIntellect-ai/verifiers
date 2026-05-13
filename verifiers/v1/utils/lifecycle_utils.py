from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Literal, cast

from verifiers.utils.async_utils import maybe_call_with_named_args

from ..state import State
from ..task import Task

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
    return sort_handlers(unique_handlers(handlers), attr)


def validate_handler_args(
    handlers: Iterable[Callable[..., object]],
    expected: set[str],
    attr: str,
    stage: LifecycleStage,
) -> None:
    expected_text = expected_args_text(expected)
    for handler in handlers:
        names = set(inspect.signature(handler).parameters)
        name = str(getattr(handler, "__name__", type(handler).__name__))
        if stage == "group" and names != expected:
            raise ValueError(
                f"group {attr} handler {name!r} must accept exactly {expected_text}."
            )
        if not expected.issubset(names):
            raise ValueError(
                f"{stage} {attr} handler {name!r} must accept {expected_text}."
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


def sort_handlers(
    handlers: Iterable[Callable[..., object]], attr: str
) -> list[Callable[..., object]]:
    return sorted(
        handlers,
        key=lambda handler: (
            -int(getattr(handler, f"{attr}_priority", 0)),
            str(getattr(handler, "__name__", "")),
        ),
    )


def expected_args_text(expected: set[str]) -> str:
    if expected == {"task", "state"}:
        return "task and state"
    if expected == {"tasks", "states"}:
        return "tasks and states"
    return " and ".join(sorted(expected))


async def state_done(task: Task, state: State) -> bool:
    _ = task
    return bool(state.get("done"))


def handler_collection_attr(attr: str) -> str:
    return {
        "stop": "stops",
        "setup": "setups",
        "update": "updates",
        "cleanup": "cleanups",
        "teardown": "teardowns",
    }.get(attr, attr)
