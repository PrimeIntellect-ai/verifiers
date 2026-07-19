"""Scoring and rollout-control decorators."""

import asyncio
import inspect
from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


def discover_decorated(obj: object, attr: str) -> list[Callable[..., Any]]:
    """Bound methods on `obj` tagged with `attr`, sorted by priority then name. Scans the
    class MRO for tagged functions (not `inspect.getmembers(obj)`, which evaluates every
    descriptor — a property with side effects would run here) and binds each through
    `getattr`, so the most-derived override wins."""
    names = {
        name
        for klass in type(obj).__mro__
        for name, fn in vars(klass).items()
        if callable(fn) and hasattr(fn, attr)
    }
    # An undecorated override suppresses a decorated base method.
    methods = [method for name in names if hasattr(method := getattr(obj, name), attr)]
    priority_attr = f"{attr}_priority"
    methods.sort(key=lambda m: (-getattr(m, priority_attr, 0), m.__name__))
    return methods


def invoke(fn: Callable[..., Any], available: dict[str, Any]) -> Any:
    params = inspect.signature(fn).parameters
    return fn(**{name: value for name, value in available.items() if name in params})


async def invoke_all(
    fns: list[Callable[..., Any]], available: dict[str, Any]
) -> list[Any]:
    """Invoke scoring handlers concurrently, including empty and singleton lists."""
    return await asyncio.gather(*(invoke(fn, available) for fn in fns))


def mark(attr: str, **extra: Any) -> Callable[[F], F]:
    def decorator(f: F) -> F:
        setattr(f, attr, True)
        for key, value in extra.items():
            setattr(f, key, value)
        return f

    return decorator


@overload
def tool(func: F, name: str | None = None) -> F: ...
@overload
def tool(func: None = None, name: str | None = None) -> Callable[[F], F]: ...
def tool(func: F | None = None, name: str | None = None) -> F | Callable[[F], F]:
    """Mark a `Toolset` method as an MCP tool exposed to the model. The tool name defaults
    to the method name (override with `name`); the docstring becomes its description."""
    decorator = mark("tool", tool_name=name)
    return decorator if func is None else decorator(func)


@overload
def stop(func: F, priority: int = 0) -> F: ...
@overload
def stop(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def stop(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a stop condition `(self, trace) -> bool`."""
    decorator = mark("stop", stop_priority=priority)
    return decorator if func is None else decorator(func)


@overload
def metric(func: F, priority: int = 0, role: str | None = None) -> F: ...
@overload
def metric(
    func: None = None, priority: int = 0, role: str | None = None
) -> Callable[[F], F]: ...
def metric(
    func: F | None = None, priority: int = 0, role: str | None = None
) -> F | Callable[[F], F]:
    """Mark a metric `(self, trace) -> float` (recorded, not summed). On an
    `Environment` it's a cross-agent signal: run once per episode trace with the
    finished sibling set in reach (`trace` = the target, `traces` = all of them);
    `role=` narrows the targets (env-only — a task has no roles)."""
    decorator = mark("metric", metric_priority=priority, _vf_role=role)
    return decorator if func is None else decorator(func)


@overload
def reward(
    func: F, weight: float = 1.0, priority: int = 0, role: str | None = None
) -> F: ...
@overload
def reward(
    func: None = None, weight: float = 1.0, priority: int = 0, role: str | None = None
) -> Callable[[F], F]: ...
def reward(
    func: F | None = None,
    weight: float = 1.0,
    priority: int = 0,
    role: str | None = None,
) -> F | Callable[[F], F]:
    """Mark a weighted per-rollout reward returning a float or keyed scores. On an
    `Environment` it's a cross-agent signal — see `metric` for the env semantics
    (`role=` picks whose traces it records onto)."""
    decorator = mark(
        "reward", reward_priority=priority, _vf_weight=weight, _vf_role=role
    )
    return decorator if func is None else decorator(func)
