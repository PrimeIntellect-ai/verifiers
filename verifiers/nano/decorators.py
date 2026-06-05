"""Decorators that program rollout control flow.

Each decorator just tags a method with attributes; `discover_decorated` collects
the tagged bound methods, sorted by descending priority then name. Copied nearly
verbatim from the (already clean) v1 `decorators.py`.

Usage on a Taskset subclass (all handlers must be `async`):

    @vf.reward(weight=1.0)
    async def my_reward(self, task, transcript) -> float: ...

    @vf.metric
    async def my_metric(self, task, transcript) -> float: ...

    @vf.setup
    async def my_setup(self, transcript) -> None: ...

    @vf.cleanup
    async def my_cleanup(self, transcript) -> None: ...

    @vf.stop
    async def my_stop(self, transcript) -> bool: ...
"""

import inspect
from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


def discover_decorated(obj: object, attr: str) -> list[Callable[..., Any]]:
    """Bound methods on `obj` tagged with `attr`, sorted by priority then name."""
    methods = [
        method
        for _, method in inspect.getmembers(obj, predicate=inspect.ismethod)
        if hasattr(method, attr)
    ]
    priority_attr = f"{attr}_priority"
    methods.sort(key=lambda m: (-getattr(m, priority_attr, 0), m.__name__))
    return methods


def mark(attr: str, **extra: Any) -> Callable[[F], F]:
    def decorator(f: F) -> F:
        setattr(f, attr, True)
        for key, value in extra.items():
            setattr(f, key, value)
        return f

    return decorator


@overload
def setup(func: F, priority: int = 0) -> F: ...
@overload
def setup(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def setup(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a rollout setup handler `(self, transcript) -> None`."""
    decorator = mark("setup", setup_priority=priority)
    return decorator if func is None else decorator(func)


@overload
def cleanup(func: F, priority: int = 0) -> F: ...
@overload
def cleanup(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def cleanup(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a rollout cleanup handler `(self, transcript) -> None` (always runs)."""
    decorator = mark("cleanup", cleanup_priority=priority)
    return decorator if func is None else decorator(func)


@overload
def stop(func: F, priority: int = 0) -> F: ...
@overload
def stop(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def stop(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a stop condition `(self, transcript) -> bool`."""
    decorator = mark("stop", stop_priority=priority)
    return decorator if func is None else decorator(func)


@overload
def metric(func: F, priority: int = 0) -> F: ...
@overload
def metric(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def metric(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a metric `(self, task, transcript) -> float` (recorded, not summed)."""
    decorator = mark("metric", metric_priority=priority)
    return decorator if func is None else decorator(func)


@overload
def reward(func: F, weight: float = 1.0, priority: int = 0) -> F: ...
@overload
def reward(
    func: None = None, weight: float = 1.0, priority: int = 0
) -> Callable[[F], F]: ...
def reward(
    func: F | None = None, weight: float = 1.0, priority: int = 0
) -> F | Callable[[F], F]:
    """Mark a reward `(self, task, transcript) -> float` (summed into transcript.reward)."""
    decorator = mark("reward", reward_priority=priority, _vf_weight=weight)
    return decorator if func is None else decorator(func)
