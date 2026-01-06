import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Generic, TypeVar, cast

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI

from verifiers.types import ClientConfig
from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)

T = TypeVar("T")


class ThreadedAsyncClient(Generic[T]):
    """
    Generic wrapper to run async clients in a threadpool.

    Used for higher performance in high-concurrency environments to alleviate
    the main event loop.

    Dispatches method calls to a ThreadPoolExecutor where
    each thread maintains its own client instance via thread-local storage.
    Supports chained attribute access (e.g., client.chat.completions.create).

    Usage:
        client = ThreadedAsyncClient(
            factory=lambda: AsyncOpenAI(base_url=..., api_key=...),
            max_workers=10,
        )
        response = await client.chat.completions.create(model="gpt-4", messages=[...])
    """

    class ChainedProxy:
        """Walks attribute path and dispatches async call to thread pool."""

        def __init__(
            self, parent: "ThreadedAsyncClient[T]", path: tuple[str, ...]
        ) -> None:
            self.parent = parent
            self.path = path

        def __getattr__(self, name: str) -> "ThreadedAsyncClient.ChainedProxy":
            return ThreadedAsyncClient.ChainedProxy(self.parent, self.path + (name,))

        async def __call__(self, *args, **kwargs) -> Any:
            def run_in_thread():
                loop = get_or_create_thread_loop()
                client = get_or_create_thread_attr(
                    f"client_{id(self.parent)}", self.parent.factory
                )
                method: Any = client
                for attr in self.path:
                    method = getattr(method, attr)
                return loop.run_until_complete(cast(Callable, method)(*args, **kwargs))

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.parent.executor, run_in_thread)

    def __init__(
        self,
        factory: Callable[[], T],
        max_workers: int = 100,
        thread_name_prefix: str = "threaded-client",
    ) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self.factory = factory

    def __getattr__(self, name: str) -> ChainedProxy:
        return self.ChainedProxy(self, (name,))

    def teardown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=wait)


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
