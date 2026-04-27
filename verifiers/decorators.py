import inspect
from typing import Any, Awaitable, Callable, Literal

SignalStage = Literal["rollout", "group"]


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


def stop(
    func: Callable[..., Awaitable[bool]] | None = None, priority: int = 0
) -> (
    Callable[..., Awaitable[bool]]
    | Callable[[Callable[..., Awaitable[bool]]], Callable[..., Awaitable[bool]]]
):
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

    def decorator(f: Callable[..., Awaitable[bool]]) -> Callable[..., Awaitable[bool]]:
        setattr(f, "stop", True)
        setattr(f, "stop_priority", priority)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


def cleanup(
    func: Callable[..., Awaitable[None]] | None = None,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> (
    Callable[..., Awaitable[None]]
    | Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]
):
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

    def decorator(f: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        setattr(f, "cleanup", True)
        setattr(f, "cleanup_priority", priority)
        setattr(f, "cleanup_stage", stage)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


def render(
    func: Callable[..., Awaitable[None]] | None = None,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> (
    Callable[..., Awaitable[None]]
    | Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]
):
    """Decorator to mark a pre-scoring state render/finalization handler."""

    def decorator(f: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        setattr(f, "render", True)
        setattr(f, "render_priority", priority)
        setattr(f, "render_stage", stage)
        return f

    if func is None:
        return decorator
    return decorator(func)


def metric(
    func: Callable[..., Awaitable[float]] | None = None,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> (
    Callable[..., Awaitable[float]]
    | Callable[[Callable[..., Awaitable[float]]], Callable[..., Awaitable[float]]]
):
    """Decorator to mark a rollout or group metric signal."""

    def decorator(
        f: Callable[..., Awaitable[float]],
    ) -> Callable[..., Awaitable[float]]:
        setattr(f, "metric", True)
        setattr(f, "metric_priority", priority)
        setattr(f, "metric_stage", stage)
        return f

    if func is None:
        return decorator
    return decorator(func)


def reward(
    func: Callable[..., Awaitable[float]] | None = None,
    weight: float = 1.0,
    priority: int = 0,
    stage: SignalStage = "rollout",
) -> (
    Callable[..., Awaitable[float]]
    | Callable[[Callable[..., Awaitable[float]]], Callable[..., Awaitable[float]]]
):
    """Decorator to mark a rollout or group reward signal."""

    def decorator(
        f: Callable[..., Awaitable[float]],
    ) -> Callable[..., Awaitable[float]]:
        setattr(f, "reward", True)
        setattr(f, "reward_priority", priority)
        setattr(f, "reward_stage", stage)
        setattr(f, "reward_weight", weight)
        return f

    if func is None:
        return decorator
    return decorator(func)


def advantage(
    func: Callable[..., Awaitable[list[float]]] | None = None,
    priority: int = 0,
) -> (
    Callable[..., Awaitable[list[float]]]
    | Callable[
        [Callable[..., Awaitable[list[float]]]], Callable[..., Awaitable[list[float]]]
    ]
):
    """Decorator to mark a group-stage advantage handler."""

    def decorator(
        f: Callable[..., Awaitable[list[float]]],
    ) -> Callable[..., Awaitable[list[float]]]:
        setattr(f, "advantage", True)
        setattr(f, "advantage_priority", priority)
        setattr(f, "advantage_stage", "group")
        return f

    if func is None:
        return decorator
    return decorator(func)


def teardown(
    func: Callable[..., Awaitable[None]] | None = None, priority: int = 0
) -> (
    Callable[..., Awaitable[None]]
    | Callable[[Callable[..., Awaitable[None]]], Callable[..., Awaitable[None]]]
):
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

    def decorator(f: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
        setattr(f, "teardown", True)
        setattr(f, "teardown_priority", priority)
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)
