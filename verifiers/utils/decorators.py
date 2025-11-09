from typing import Awaitable, Callable


def stop(func: Callable[..., Awaitable[bool]]) -> Callable[..., Awaitable[bool]]:
    """
    Decorator to mark a method as a stop condition.

    The decorated function should take a State and return a bool (or Awaitable[bool]).
    All stop conditions are automatically checked by is_completed.
    """
    setattr(func, "stop", True)
    return func


def cleanup(func: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """
    Decorator to mark a method as a rollout cleanup.

    The decorated function should take a State and return an Awaitable[None].
    All cleanup functions are automatically called by rollout.
    """
    setattr(func, "cleanup", True)
    return func


def teardown(func: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """
    Decorator to mark a method as a teardown handler.

    The decorated Environment method should return an Awaitable[None].
    All teardown handlers are automatically when the environment is destroyed.
    """
    setattr(func, "teardown", True)
    return func
