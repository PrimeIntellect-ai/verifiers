"""Decorators that program rollout control flow + scoring.

Each decorator just tags a method with attributes; `discover_decorated` collects
the tagged bound methods, sorted by descending priority then name.

Scoring methods declare the inputs they need *by parameter name* and the framework
injects them (`invoke`) — so a pure-trace reward needn't take the runtime, and an
in-runtime verifier needn't take the trace. They live on the `Task` subclass, so
`self` IS the task; the available names are:

    @reward / @metric  (per rollout):  trace, runtime
    @group_reward      (per group):    traces

All handlers must be `async`. Examples (on a Task subclass):

    @vf.reward(weight=1.0)
    async def correct(self, trace, runtime) -> float: ...         # both

    @vf.metric
    async def turns(self, trace) -> float: ...                    # trace only

    @vf.group_reward
    async def most_concise(self, traces) -> list[float]: ...      # compares this task's rollouts

A `@group_reward` compares trace metadata across a task's rollouts; anything from the
runtime is recorded per rollout as a `@metric` first.
"""

import inspect
from typing import Any, Callable, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


def discover_decorated(obj: object, attr: str) -> list[Callable[..., Any]]:
    """Bound methods on `obj` tagged with `attr`, sorted by priority then name. Scans the
    class MRO for tagged functions (not `inspect.getmembers(obj)`, which evaluates every
    descriptor — e.g. a property like `Task.config` that raises until attached) and binds
    each through `getattr`, so the most-derived override wins."""
    names = {
        name
        for klass in type(obj).__mro__
        for name, fn in vars(klass).items()
        if callable(fn) and hasattr(fn, attr)
    }
    methods = [getattr(obj, name) for name in names]
    priority_attr = f"{attr}_priority"
    methods.sort(key=lambda m: (-getattr(m, priority_attr, 0), m.__name__))
    return methods


def invoke(fn: Callable[..., Any], available: dict[str, Any]) -> Any:
    """Call `fn`, passing only the items of `available` whose name it declares as a
    parameter. `fn` is a bound method (so `self` is already excluded); the result is
    a coroutine because handlers are async."""
    params = inspect.signature(fn).parameters
    return fn(**{name: value for name, value in available.items() if name in params})


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
    """Mark a per-rollout reward (weighted, summed into trace.reward). Declare any of
    `trace`/`runtime`; they're injected by name (`self` is the task). Return a `float`
    (recorded under the method name) or a `dict[str, float]` (each entry under its own
    key); every contribution is scaled by `weight` before it's summed into `trace.reward`."""
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
    """Mark a group reward `(self, traces) -> list[float]`: one score per trace,
    computed by comparing all rollouts of this task (e.g. pairwise preference) — a function
    of trace metadata. Each score is weighted and summed into that trace's reward,
    alongside the per-rollout rewards."""
    decorator = mark("group_reward", group_reward_priority=priority, _vf_weight=weight)
    return decorator if func is None else decorator(func)
