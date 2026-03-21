import asyncio
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, cast

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

# Default scaling: 1:1 concurrency to max_workers
ScalingFn = Callable[[int], int]


def _default_scaling(concurrency: int) -> int:
    return concurrency


Executor = ThreadPoolExecutor | ProcessPoolExecutor
_executor_registry: dict[str, tuple[Executor, ScalingFn]] = {}
_default_executor: ThreadPoolExecutor | None = None
_target_concurrency: int | None = None  # sticky target from last scale_executors call


def _resize(executor: Executor, max_workers: int) -> None:
    """Resize an executor in-place. Workers are spawned lazily so
    raising the limit simply allows more workers on the next submit."""
    cast(Any, executor)._max_workers = max_workers


def register_executor(
    name: str,
    executor: Executor,
    scaling_fn: ScalingFn | None = None,
) -> None:
    """Register an executor so it is resized by future :func:`scale_executors` calls.

    *scaling_fn* maps concurrency → max_workers for this executor.
    Defaults to 1:1 if not provided.

    If :func:`scale_executors` was already called, the executor is immediately
    resized using the scaling function.
    """
    fn = scaling_fn or _default_scaling
    _executor_registry[name] = (executor, fn)

    if _target_concurrency is not None:
        target = max(1, fn(_target_concurrency))
        if cast(Any, executor)._max_workers != target:
            _resize(executor, target)
            logger.debug(
                f"Registered executor {name} and immediately scaled to "
                f"max_workers={target} (concurrency={_target_concurrency})"
            )
            return
    logger.debug(
        f"Registered executor {name} (max_workers={cast(Any, executor)._max_workers})"
    )


def unregister_executor(name: str) -> None:
    """Remove a previously registered executor (does **not** shut it down)."""
    _executor_registry.pop(name, None)


def scale_executors(concurrency: int) -> int:
    """Scale the default event-loop executor **and** all registered executors.

    Each registered executor applies its own scaling function to map
    *concurrency* to a max_workers value (default 1:1).

    If a running event loop exists, the default executor is bound to it
    immediately.  Otherwise the executor is only created/resized and the
    caller must call :func:`install_default_executor` once inside the real
    loop (e.g. at the start of ``async def run()``).

    Returns *concurrency*.
    """
    global _default_executor, _target_concurrency

    _target_concurrency = concurrency

    # default event-loop executor: 1:1 scaling
    if _default_executor is None:
        _default_executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="vf-default"
        )
    else:
        _resize(_default_executor, concurrency)

    # If there is already a running loop, bind immediately. When called
    # outside of an async context (e.g. during __init__ before asyncio.run),
    # this is a no-op and the caller must use install_default_executor() later.
    try:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(_default_executor)
    except RuntimeError:
        pass  # no running loop yet — caller must call install_default_executor()

    # explicitly registered executors — each applies its own scaling
    targets = []
    for name, (executor, scaling_fn) in _executor_registry.items():
        target = max(1, scaling_fn(concurrency))
        _resize(executor, target)
        logger.debug(f"Scaled executor {name} to max_workers={target} ({concurrency=})")
        targets.append(target)

    targets_str = ", ".join(
        f"{n}={t}" for n, t in zip(_executor_registry.keys(), targets)
    )
    logger.info(
        f"Scaled {len(_executor_registry)} registered executor(s) and default executor ({targets_str})"
    )
    return concurrency


def install_default_executor() -> None:
    """Bind the default executor to the **currently running** event loop.

    Call this early inside an ``async`` function (after ``asyncio.run()`` has
    created the real loop) so that ``run_in_executor(None, ...)`` uses the
    scaled thread pool.  Safe to call multiple times — it is a no-op if no
    default executor has been created yet.
    """
    if _default_executor is not None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(_default_executor)
        logger.debug(
            f"Installed default executor (max_workers={_default_executor._max_workers}) "
            f"on loop {id(loop)}"
        )


def shutdown_executors() -> None:
    """Shut down the default executor and all registered executors."""
    global _default_executor, _target_concurrency
    _target_concurrency = None
    if _default_executor is not None:
        _default_executor.shutdown(wait=False)
        _default_executor = None
    for executor, _ in _executor_registry.values():
        executor.shutdown(wait=False)
    _executor_registry.clear()
