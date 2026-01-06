import asyncio
import os
import threading
from typing import Any, Callable, Generic, TypeVar

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig

T = TypeVar("T")


class ThreadedAsyncClient(Generic[T]):
    """
    Generic wrapper to run async clients in a threadpool.

    Used for higher performance in high-concurrency environments to alleviate
    the main event loop.

    Each worker thread runs its own event loop via run_forever(), allowing
    multiple concurrent async requests per worker. Coroutines are submitted
    via run_coroutine_threadsafe and results are awaited in the main loop.

    Usage:
        client = ThreadedAsyncClient(
            factory=lambda: AsyncOpenAI(base_url=..., api_key=...),
            max_workers=10,
        )
        response = await client.chat.completions.create(model="gpt-4", messages=[...])
    """

    class ChainedProxy:
        """Walks attribute path and dispatches async call to worker loop."""

        def __init__(
            self, parent: "ThreadedAsyncClient[T]", path: tuple[str, ...]
        ) -> None:
            self.parent = parent
            self.path = path

        def __getattr__(self, name: str) -> "ThreadedAsyncClient.ChainedProxy":
            return ThreadedAsyncClient.ChainedProxy(self.parent, self.path + (name,))

        async def __call__(self, *args, **kwargs) -> Any:
            loop, client = self.parent._get_worker()
            method: Any = client
            for attr in self.path:
                method = getattr(method, attr)
            future = asyncio.run_coroutine_threadsafe(method(*args, **kwargs), loop)
            return await asyncio.wrap_future(future)

    def __init__(
        self,
        factory: Callable[[], T],
        max_workers: int = 100,
        thread_name_prefix: str = "threaded-client",
    ) -> None:
        self.factory = factory
        self._workers: list[tuple[asyncio.AbstractEventLoop, T]] = []
        self._lock = threading.Lock()
        self._idx = 0
        self._ready = threading.Barrier(max_workers + 1)

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
        with self._lock:
            self._workers.append((loop, client))
        self._ready.wait()
        loop.run_forever()

    def _get_worker(self) -> tuple[asyncio.AbstractEventLoop, Any]:
        with self._lock:
            worker = self._workers[self._idx]
            self._idx = (self._idx + 1) % len(self._workers)
        return worker

    def __getattr__(self, name: str) -> ChainedProxy:
        return self.ChainedProxy(self, (name,))

    def teardown(self) -> None:
        """Stop worker loops."""
        for loop, _ in self._workers:
            loop.call_soon_threadsafe(loop.stop)


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
