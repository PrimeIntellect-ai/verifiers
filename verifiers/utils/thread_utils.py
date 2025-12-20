import asyncio
import threading
from typing import Any, Callable

THREAD_LOCAL_STORAGE = threading.local()


def get_thread_local_storage() -> threading.local:
    """Get the thread-local storage for the current thread."""
    return THREAD_LOCAL_STORAGE


def get_or_create_thread_attr(
    key: str, factory: Callable[..., Any], *args, **kwargs
) -> Any:
    """Get value from thread-local storage, creating it if it doesn't exist."""
    thread_local = get_thread_local_storage()
    value = getattr(thread_local, key, None)
    if value is None:
        value = factory(*args, **kwargs)
        setattr(thread_local, key, value)
    return value


def get_or_create_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create event loop for current thread. Reuses loop to avoid closing it."""
    thread_local_loop = get_or_create_thread_attr("loop", asyncio.new_event_loop)
    asyncio.set_event_loop(thread_local_loop)
    return thread_local_loop
