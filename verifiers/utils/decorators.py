from typing import Awaitable, Callable


def stop(
    func: Callable[..., Awaitable[bool]] | None = None, *, priority: int = 0
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
    func: Callable[..., Awaitable[None]] | None = None, *, priority: int = 0
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
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)


def teardown(
    func: Callable[..., Awaitable[None]] | None = None, *, priority: int = 0
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
