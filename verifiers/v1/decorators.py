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
    methods = [
        (name, method) for name in names if hasattr(method := getattr(obj, name), attr)
    ]
    priority_attr = f"{attr}_priority"
    methods.sort(key=lambda item: (-getattr(item[1], priority_attr, 0), item[0]))
    return [method for _, method in methods]


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
def intercept(func: F, priority: int = 0) -> F: ...
@overload
def intercept(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def intercept(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a `Task` method as a message interceptor.

    Interceptors may declare `message` and `trace`. They see tool messages before the model
    does and assistant messages before the harness does; the `message` annotation selects
    which (`vf.AssistantMessage`, `vf.ToolMessage`, or unannotated for both). Return None to
    pass through the native message untouched, a string to replace an assistant turn (its tool
    calls never run) or a tool result's content, or a typed replacement message. The first
    replacement wins.

        block_rm = vf.block_shell_commands("rm")

    A replacement is what the harness sees and what the trace records. Its own provider state and
    reasoning are discarded; dialects may retain original continuation state required by a
    rewritten tool call. Tool-message interceptors re-run on the same message every turn — the
    harness replays the original history — so they must be deterministic. Streamed responses are
    withheld until the interceptors have ruled.
    """
    decorator = mark("intercept", intercept_priority=priority)
    return decorator if func is None else decorator(func)


@overload
def metric(func: F, priority: int = 0) -> F: ...
@overload
def metric(func: None = None, priority: int = 0) -> Callable[[F], F]: ...
def metric(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a metric `(self, trace) -> float` (recorded, not summed)."""
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
    """Mark a weighted per-rollout reward returning a float or keyed scores."""
    decorator = mark("reward", reward_priority=priority, _vf_weight=weight)
    return decorator if func is None else decorator(func)


@overload
def group_reward(func: F, weight: float = 1.0, priority: int = 0) -> F: ...
@overload
def group_reward(
    func: None = None, weight: float = 1.0, priority: int = 0
) -> Callable[[F], F]: ...
def group_reward(
    func: F | None = None, weight: float = 1.0, priority: int = 0
) -> F | Callable[[F], F]:
    """Mark a weighted group reward returning one score per trace."""
    decorator = mark("group_reward", group_reward_priority=priority, _vf_weight=weight)
    return decorator if func is None else decorator(func)
