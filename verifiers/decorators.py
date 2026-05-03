import inspect
from typing import Any, Callable, Literal, TypeVar, overload

SignalStage = Literal["rollout", "group"]
F = TypeVar("F", bound=Callable[..., object])


def discover_decorated(obj: Any, attr: str) -> list:
    """Discover methods decorated with a given attribute, sorted by priority.

    Returns bound methods on *obj* that have ``attr`` set, ordered by
    descending ``{attr}_priority`` then ascending ``__name__``.
    """
    methods = [
        method
        for _, method in inspect.getmembers(obj, predicate=inspect.ismethod)
        if hasattr(method, attr) and callable(method)
    ]
    priority_attr = f"{attr}_priority"
    methods.sort(key=lambda m: (-getattr(m, priority_attr, 0), m.__name__))
    return methods


@overload
def stop(func: F, priority: int = 0) -> F: ...


@overload
def stop(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def stop(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """
    Decorator to mark a method as a stop condition.

    The decorated function should take a State and return a bool (or Awaitable[bool]).
    All stop conditions are automatically checked by is_completed.

    Args:
        func: The function to decorate (when used as @stop)
        priority: Optional priority to control execution order. Defaults to 0.
            Higher priorities run first. Use higher numbers to run earlier, lower numbers to run later.
            Ties are broken alphabetically by function name.

    Examples:
        @vf.stop
        async def my_stop_condition(self, state: State) -> bool:
            ...

        @vf.stop(priority=10)
        async def early_check(self, state: State) -> bool:
            ...

        @vf.stop(priority=-5)
        async def late_check(self, state: State) -> bool:
            ...
    """

    def decorator(f: F) -> F:
        setattr(f, "stop", True)
        setattr(f, "stop_priority", priority)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


@overload
def cleanup(func: F, priority: int = 0, stage: SignalStage = "rollout") -> F: ...


@overload
def cleanup(
    func: None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> Callable[[F], F]: ...


def cleanup(
    func: F | None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> F | Callable[[F], F]:
    """
    Decorator to mark a method as a rollout cleanup.

    The decorated function should take a State and return an Awaitable[None].
    All cleanup functions are automatically called by rollout.

    Args:
        func: The function to decorate (when used as @cleanup)
        priority: Optional priority to control execution order. Defaults to 0.
            Higher priorities run first. Use higher numbers to run earlier, lower numbers to run later.
            Ties are broken alphabetically by function name.
    Examples:
        @vf.cleanup
        async def my_cleanup(self, state: State):
            ...

        @vf.cleanup(priority=10)
        async def early_cleanup(self, state: State):
            ...

        @vf.cleanup(priority=-5)
        async def late_cleanup(self, state: State):
            ...
    """

    def decorator(f: F) -> F:
        setattr(f, "cleanup", True)
        setattr(f, "cleanup_priority", priority)
        setattr(f, "cleanup_stage", stage)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


@overload
def metric(func: F, priority: int = 0, stage: SignalStage = "rollout") -> F: ...


@overload
def metric(
    func: None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> Callable[[F], F]: ...


def metric(
    func: F | None = None, priority: int = 0, stage: SignalStage = "rollout"
) -> F | Callable[[F], F]:
    """Decorator to mark a rollout or group metric signal."""

    def decorator(f: F) -> F:
        setattr(f, "metric", True)
        setattr(f, "metric_priority", priority)
        setattr(f, "metric_stage", stage)
        return f

    if func is None:
        return decorator
    return decorator(func)


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
    """Decorator to mark a rollout or group reward signal."""

    def decorator(f: F) -> F:
        setattr(f, "reward", True)
        setattr(f, "reward_priority", priority)
        setattr(f, "reward_stage", stage)
        setattr(f, "reward_weight", weight)
        return f

    if func is None:
        return decorator
    return decorator(func)


@overload
def advantage(func: F, priority: int = 0) -> F: ...


@overload
def advantage(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def advantage(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Decorator to mark a group-stage advantage handler."""

    def decorator(f: F) -> F:
        setattr(f, "advantage", True)
        setattr(f, "advantage_priority", priority)
        setattr(f, "advantage_stage", "group")
        return f

    if func is None:
        return decorator
    return decorator(func)


@overload
def teardown(func: F, priority: int = 0) -> F: ...


@overload
def teardown(func: None = None, priority: int = 0) -> Callable[[F], F]: ...


def teardown(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """
    Decorator to mark a method as a teardown handler.

    The decorated Environment method should return an Awaitable[None].
    All teardown handlers are automatically when the environment is destroyed.

    Args:
        func: The function to decorate (when used as @teardown)
        priority: Optional priority to control execution order. Defaults to 0.
            Higher priorities run first. Use higher numbers to run earlier, lower numbers to run later.
            Ties are broken alphabetically by function name.

    Examples:
        @vf.teardown
        async def my_teardown(self):
            ...

        @vf.teardown(priority=10)
        async def early_teardown(self):
            ...

        @vf.teardown(priority=-5)
        async def late_teardown(self):
            ...
    """

    def decorator(f: F) -> F:
        setattr(f, "teardown", True)
        setattr(f, "teardown_priority", priority)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)
