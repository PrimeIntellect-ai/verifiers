import asyncio
import itertools
import threading
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Generic, TypeVar

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


T = TypeVar("T")


class _Threaded(Generic[T]):
    """
    Generic wrapper to run class replicas in a threadpool. Used to improve
    performance in high-concurrency environments to remove pressure from the
    main event loop. Used mostly for wrapping async clients (AsyncOpenAI,
    AsyncSandboxClient, etc.) when doing 1k+ requests.

    Creates max_workers threads, each running their own event loop with a
    thread-local replica of the class. ThreadedClass proxies the class by
    round-robin dispatching to the workers. Because each worker owns its own
    event loop, it can run multiple concurrent requests per worker.

    The wrapper supports:
        - Class + instance attribute access (via a reference instance)
        - Sync method calls (executed on reference instance)
        - Async method calls (dispatched to worker threads)

    Uses a TYPE_CHECKING trick so that ThreadedClass[T] appears as T
    to static type checkers, enabling full autocomplete and type checking.

    Example Usage:
        threaded_client = ThreadedClass(
            factory=lambda: AsyncOpenAI(base_url=..., api_key=...),
            max_workers=10,
        )
        response = await threaded_client.chat.completions.create(model="gpt-4", messages=[...])
    """

    class ChainedProxy:
        """Walks attribute path and dispatches calls to worker loop."""

        def __init__(self, parent: "_Threaded[T]", path: tuple[str, ...]):
            self.parent = parent
            self.path = path

        def __getattr__(self, name: str) -> "_Threaded.ChainedProxy":
            return _Threaded.ChainedProxy(self.parent, self.path + (name,))

        def __call__(self, *args, **kwargs) -> Any | Coroutine[Any, Any, Any]:
            def resolve_path(client: Any) -> Any:
                """Walk the attribute path to get the target."""
                target = client
                for attr in self.path:
                    target = getattr(target, attr)
                return target

            ref_method = resolve_path(self.parent._ref)
            if asyncio.iscoroutinefunction(ref_method):
                # async: return a coroutine that dispatches to worker
                async def async_dispatch() -> Any:
                    def get_worker() -> tuple[asyncio.AbstractEventLoop, Any]:
                        idx = next(self.parent._counter) % len(self.parent._workers)
                        return self.parent._workers[idx]

                    loop, client = get_worker()
                    method = resolve_path(client)
                    future = asyncio.run_coroutine_threadsafe(
                        method(*args, **kwargs), loop
                    )
                    return await asyncio.wrap_future(future)

                return async_dispatch()
            else:
                # sync: execute on reference client directly
                return ref_method(*args, **kwargs)

    def __init__(
        self,
        factory: Callable[[], T],
        max_workers: int = 100,
        thread_name_prefix: str = "threaded-class",
    ) -> None:
        self.factory = factory
        self._workers: list[tuple[asyncio.AbstractEventLoop, T]] = []
        self._init_lock = threading.Lock()
        self._counter = itertools.count()
        self._ready = threading.Barrier(max_workers + 1)
        self._ref = self.factory()

        def start_event_loop_in_thread() -> None:
            """Run a new event loop in a background thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = self.factory()
            with self._init_lock:
                self._workers.append((loop, client))
            self._ready.wait()
            loop.run_forever()

        # spawn workers as daemons so they auto-terminate when main program exits
        for i in range(max_workers):
            t = threading.Thread(
                target=start_event_loop_in_thread,
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()

        # wait for all workers to be ready
        self._ready.wait()

    def __getattr__(self, name: str) -> Any:
        # Get attribute from reference client
        ref_attr = getattr(self._ref, name)

        # return if non-callable
        if not callable(ref_attr):
            return ref_attr

        # else, return a proxy to handle the call
        return self.ChainedProxy(self, (name,))


if TYPE_CHECKING:
    # For static type checking, pretend ThreadedAsyncClient returns the wrapped client directly.
    # This enables IDE autocomplete and type checking for the wrapped client's API.
    class Threaded(Generic[T]):
        def __new__(
            cls,
            factory: Callable[[], T],
            max_workers: int = 10,
            thread_name_prefix: str = "threaded-class",
        ) -> T: ...
else:
    Threaded = _Threaded
