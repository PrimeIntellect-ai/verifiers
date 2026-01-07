import asyncio
import itertools
import os
import threading
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Generic, TypeVar

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig

T = TypeVar("T")


class _ThreadedAsyncClient(Generic[T]):
    """
    Generic wrapper to run async clients in a threadpool.

    Used for higher performance in high-concurrency environments to alleviate
    the main event loop.

    Each worker thread runs its own event loop via run_forever(), allowing
    multiple concurrent async requests per worker. Coroutines are submitted
    via run_coroutine_threadsafe and results are awaited in the main loop.

    Supports:
        - Regular attribute access (e.g., client.base_url)
        - Sync method calls (e.g., client.copy())
        - Async method calls dispatched to worker threads

    Typing:
        Uses a TYPE_CHECKING trick so that ThreadedAsyncClient[T] appears as T
        to static type checkers, enabling full autocomplete and type checking
        for the wrapped client's API.

    Usage:
        client = ThreadedAsyncClient(
            factory=lambda: AsyncOpenAI(base_url=..., api_key=...),
            max_workers=10,
        )
        response = await client.chat.completions.create(model="gpt-4", messages=[...])
    """

    class ChainedProxy:
        """Walks attribute path and dispatches calls to worker loop."""

        def __init__(
            self, parent: "_ThreadedAsyncClient[T]", path: tuple[str, ...]
        ) -> None:
            self.parent = parent
            self.path = path

        def _resolve_path(self, client: Any) -> Any:
            """Walk the attribute path to get the target."""
            target = client
            for attr in self.path:
                target = getattr(target, attr)
            return target

        def __getattr__(self, name: str) -> "_ThreadedAsyncClient.ChainedProxy":
            return _ThreadedAsyncClient.ChainedProxy(self.parent, self.path + (name,))

        def __call__(self, *args, **kwargs) -> Any | Coroutine[Any, Any, Any]:
            # Check on reference client whether this is async or sync
            ref_method = self._resolve_path(self.parent._ref)

            if asyncio.iscoroutinefunction(ref_method):
                # Async method - return a coroutine that dispatches to worker
                async def async_dispatch() -> Any:
                    loop, client = self.parent._get_worker()
                    method = self._resolve_path(client)
                    future = asyncio.run_coroutine_threadsafe(
                        method(*args, **kwargs), loop
                    )
                    return await asyncio.wrap_future(future)

                return async_dispatch()
            else:
                # Sync method - execute on reference client directly
                return ref_method(*args, **kwargs)

    def __init__(
        self,
        factory: Callable[[], T],
        max_workers: int = 100,
        thread_name_prefix: str = "threaded-client",
    ) -> None:
        self.factory = factory
        self._workers: list[tuple[asyncio.AbstractEventLoop, T]] = []
        self._init_lock = threading.Lock()
        self._counter = itertools.count()
        self._ready = threading.Barrier(max_workers + 1)
        self._ref = self.factory()

        # Spawn daemon threads - they auto-terminate when main program exits
        for i in range(max_workers):
            t = threading.Thread(
                target=self._run_worker,
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()

        # Wait for all workers to initialize
        self._ready.wait()

    def _run_worker(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.factory()
        with self._init_lock:
            self._workers.append((loop, client))
        self._ready.wait()
        loop.run_forever()

    def _get_worker(self) -> tuple[asyncio.AbstractEventLoop, Any]:
        idx = next(self._counter) % len(self._workers)
        return self._workers[idx]

    def __getattr__(self, name: str) -> Any:
        # Get attribute from reference client
        ref_attr = getattr(self._ref, name)

        # Non-callable attributes: return value directly
        if not callable(ref_attr):
            return ref_attr

        # Callable: return a proxy to handle the call
        return self.ChainedProxy(self, (name,))


if TYPE_CHECKING:
    # For static type checking, pretend ThreadedAsyncClient returns the wrapped client directly.
    # This enables IDE autocomplete and type checking for the wrapped client's API.
    class ThreadedAsyncClient(Generic[T]):
        def __new__(
            cls,
            factory: Callable[[], T],
            max_workers: int = 100,
            thread_name_prefix: str = "threaded-client",
        ) -> T: ...

        def teardown(self) -> None: ...
else:
    ThreadedAsyncClient = _ThreadedAsyncClient


def setup_client(
    config: ClientConfig,
) -> AsyncOpenAI:
    """
    A helper function to setup an AsyncOpenAI client.
    """
    # Setup timeouts and limits
    http_timeout = httpx.Timeout(config.timeout, connect=5.0)
    limits = httpx.Limits(
        max_connections=config.max_connections,
        max_keepalive_connections=config.max_keepalive_connections,
    )

    # Setup client
    http_client = AsyncClient(
        limits=limits,
        timeout=http_timeout,
        headers=config.extra_headers,
    )
    client = AsyncOpenAI(
        base_url=config.api_base_url,
        api_key=os.getenv(config.api_key_var, "EMPTY"),
        max_retries=config.max_retries,
        http_client=http_client,
    )

    return client
