import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)

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


# --- Executor registry & scaling ---

_executor_registry: dict[str, ThreadPoolExecutor] = {}
_default_executor: ThreadPoolExecutor | None = None
_target_max_workers: int | None = None  # sticky target from last scale_executors call


def _resize(executor: ThreadPoolExecutor, max_workers: int) -> None:
    """Resize a ThreadPoolExecutor in-place. Threads are spawned lazily so
    raising the limit simply allows more threads on the next submit."""
    executor._max_workers = max_workers


def register_executor(name: str, executor: ThreadPoolExecutor) -> None:
    """Register an executor so it is resized by future :func:`scale_executors` calls.

    If :func:`scale_executors` was already called, the executor is immediately
    resized to match the active target.
    """
    _executor_registry[name] = executor

    if _target_max_workers is not None and executor._max_workers != _target_max_workers:
        _resize(executor, _target_max_workers)
        logger.debug(
            f"Registered executor {name} and immediately scaled to "
            f"max_workers={_target_max_workers}"
        )
    else:
        logger.debug(
            f"Registered executor {name} (max_workers={executor._max_workers})"
        )


def unregister_executor(name: str) -> None:
    """Remove a previously registered executor (does **not** shut it down)."""
    _executor_registry.pop(name, None)


def recommended_max_workers(concurrency: int, cap: int = 4096) -> int:
    """Return a max_workers value scaled to *concurrency*.

    For I/O-bound workloads (API calls, sandbox RPCs) the thread count can
    safely far exceed the CPU count since threads spend most of their time
    blocked on network I/O.  The *cap* is a sanity limit to prevent
    misconfiguration from exhausting memory (~8 MB stack per thread).
    """
    return max(1, min(concurrency, cap))


def scale_executors(max_workers: int) -> int:
    """Scale the default event-loop executor **and** all registered executors.

    The default event-loop executor is *always* set (it does not need to be
    registered).  Registered executors are resized in-place.

    Returns *max_workers*.
    """
    global _default_executor, _target_max_workers

    _target_max_workers = max_workers

    # default event-loop executor (always tracked)
    if _default_executor is None:
        _default_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="vf-default"
        )
        loop = asyncio.get_event_loop()
        loop.set_default_executor(_default_executor)
    else:
        _resize(_default_executor, max_workers)

    # explicitly registered executors
    for name, executor in _executor_registry.items():
        _resize(executor, max_workers)
        logger.debug(f"Scaled executor {name} to max_workers={max_workers}")

    logger.info(
        f"scale_executors({max_workers}): default + {len(_executor_registry)} registered executor(s)"
    )
    return max_workers


def shutdown_executors() -> None:
    """Shut down the default executor and all registered executors."""
    global _default_executor, _target_max_workers
    _target_max_workers = None
    if _default_executor is not None:
        _default_executor.shutdown(wait=False)
        _default_executor = None
    for executor in _executor_registry.values():
        executor.shutdown(wait=False)
    _executor_registry.clear()
