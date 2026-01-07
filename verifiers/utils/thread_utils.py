import asyncio
import itertools
import threading
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

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
    Generic wrapper to dispatch operations to thread-local class replicas.

    Designed for high-concurrency async clients (AsyncOpenAI, AsyncSandboxClient, etc.)
    where 1k+ concurrent requests would overwhelm a single event loop.

    Creates max_workers threads, each running their own event loop with a
    thread-local replica of the class. Acts as an exact mirror of the wrapped class:
    - Async methods return awaitables
    - Sync methods return values directly
    - Attributes return values directly
    All attributes and methods may be nested (i.e. `client.chat.completions.create(...)`)

    Example Usage:
        client = Threaded(factory=lambda: AsyncOpenAI(...), max_workers=10)
        response = await client.chat.completions.create(model="gpt-4", messages=[...])
        base_url = client.base_url  # Direct attribute access
    """

    class ChainedProxy:
        """Accumulates attribute path and dispatches method calls to workers."""

        __slots__ = ("_parent", "_path")
        _parent: "_Threaded"
        _path: tuple[str, ...]

        def __init__(self, parent: "_Threaded[T]", path: tuple[str, ...]):
            object.__setattr__(self, "_parent", parent)
            object.__setattr__(self, "_path", path)

        def _resolve_path(self, root: T) -> Any:
            """Resolve the attribute path of a root instance."""
            target = root
            for attr in self._path:
                target = getattr(target, attr)
            return target

        def __getattr__(self, name: str) -> "_Threaded.ChainedProxy | Any":
            """Extend path or resolve to value if target is not callable."""
            new_path = self._path + (name,)
            is_callable, value = self._parent._resolve_path_callable(new_path)
            if is_callable:
                return _Threaded.ChainedProxy(self._parent, new_path)
            return value

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Dispatch method call to worker, returning awaitable for async methods."""
            loop, root = self._parent._get_worker()

            async def check_is_async() -> bool:
                target = self._resolve_path(root)
                return asyncio.iscoroutinefunction(target)

            future = asyncio.run_coroutine_threadsafe(check_is_async(), loop)
            is_async = future.result()

            async def execute() -> Any:
                target = self._resolve_path(root)
                result = target(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            if is_async:

                async def dispatch() -> Any:
                    f = asyncio.run_coroutine_threadsafe(execute(), loop)
                    return await asyncio.wrap_future(f)

                return dispatch()
            else:
                future = asyncio.run_coroutine_threadsafe(execute(), loop)
                return future.result()

    def __init__(
        self,
        factory: Callable[[], T],
        max_workers: int = 100,
        thread_name_prefix: str = "threaded",
    ) -> None:
        self._factory = factory
        self._workers: list[tuple[asyncio.AbstractEventLoop, T]] = []
        self._init_lock = threading.Lock()
        self._counter = itertools.count()
        self._ready = threading.Barrier(max_workers + 1)

        def start_worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            root = factory()
            with self._init_lock:
                self._workers.append((loop, root))
            self._ready.wait()
            loop.run_forever()

        for i in range(max_workers):
            t = threading.Thread(
                target=start_worker,
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()

        self._ready.wait()

    def _get_worker(self) -> tuple[asyncio.AbstractEventLoop, T]:
        """Get next worker in round-robin fashion."""
        idx = next(self._counter) % len(self._workers)
        return self._workers[idx]

    def _resolve_path_callable(self, path: tuple[str, ...]) -> tuple[bool, Any]:
        """Resolve path on worker and return (is_callable, value)."""
        loop, root = self._get_worker()

        async def resolve() -> tuple[bool, Any]:
            target = root
            for attr in path:
                target = getattr(target, attr)
            return callable(target), target

        future = asyncio.run_coroutine_threadsafe(resolve(), loop)
        return future.result()

    def __getattr__(self, name: str) -> "ChainedProxy | Any":
        """Access attribute, returning ChainedProxy for callables, value otherwise."""
        path = (name,)
        is_callable, value = self._resolve_path_callable(path)
        if is_callable:
            return self.ChainedProxy(self, path)
        return value


if TYPE_CHECKING:

    class Threaded(Generic[T]):
        def __new__(
            cls,
            factory: Callable[[], T],
            max_workers: int = 10,
            thread_name_prefix: str = "threaded-class",
        ) -> T: ...
else:
    Threaded = _Threaded
