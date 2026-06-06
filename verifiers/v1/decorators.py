from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal, TypeVar, overload

SignalStage = Literal["rollout", "group"]
F = TypeVar("F", bound=Callable[..., object])


def discover_decorated(obj: object, attr: str) -> list[Callable[..., object]]:
    methods = [
        method
        for _, method in inspect.getmembers(obj, predicate=inspect.ismethod)
        if hasattr(method, attr) and callable(method)
    ]
    priority_attr = f"{attr}_priority"
    return sorted(
        methods,
        key=lambda method: (-getattr(method, priority_attr, 0), method.__name__),
    )


def _mark(
    func: F | None,
    *,
    attr: str,
    priority: int,
    stage: SignalStage | None = None,
    weight: float | None = None,
) -> F | Callable[[F], F]:
    def decorator(f: F) -> F:
        setattr(f, attr, True)
        setattr(f, f"{attr}_priority", priority)
        if stage is not None:
            setattr(f, f"{attr}_stage", stage)
        if weight is not None:
            setattr(f, f"{attr}_weight", weight)
        return f

    return decorator if func is None else decorator(func)


@overload
def stop(func: F, priority: int = 0) -> F: ...


@overload
def stop(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def stop(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    return _mark(func, attr="stop", priority=priority)


@overload
def setup(func: F, priority: int = 0) -> F: ...


@overload
def setup(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def setup(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    return _mark(func, attr="setup", priority=priority)


@overload
def cleanup(func: F, priority: int = 0, stage: SignalStage = "rollout") -> F: ...


@overload
def cleanup(
    func: None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> Callable[[F], F]: ...


def cleanup(
    func: F | None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> F | Callable[[F], F]:
    return _mark(func, attr="cleanup", priority=priority, stage=stage)


@overload
def update(func: F, priority: int = 0, stage: SignalStage = "rollout") -> F: ...


@overload
def update(
    func: None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> Callable[[F], F]: ...


def update(
    func: F | None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> F | Callable[[F], F]:
    return _mark(func, attr="update", priority=priority, stage=stage)


@overload
def metric(func: F, priority: int = 0, stage: SignalStage = "rollout") -> F: ...


@overload
def metric(
    func: None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> Callable[[F], F]: ...


def metric(
    func: F | None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> F | Callable[[F], F]:
    return _mark(func, attr="metric", priority=priority, stage=stage)


@overload
def reward(
    func: F,
    weight: float = 1.0,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> F: ...


@overload
def reward(
    func: None = None,
    weight: float = 1.0,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> Callable[[F], F]: ...


def reward(
    func: F | None = None,
    weight: float = 1.0,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> F | Callable[[F], F]:
    return _mark(func, attr="reward", priority=priority, stage=stage, weight=weight)


@overload
def advantage(func: F, priority: int = 0) -> F: ...


@overload
def advantage(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def advantage(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    return _mark(func, attr="advantage", priority=priority, stage="group")


@overload
def teardown(func: F, priority: int = 0) -> F: ...


@overload
def teardown(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def teardown(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    return _mark(func, attr="teardown", priority=priority)
