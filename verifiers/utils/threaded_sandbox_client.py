import asyncio
import functools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from prime_sandboxes import AsyncSandboxClient, CommandTimeoutError

from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
    get_thread_local_storage,
    register_executor,
    unregister_executor,
)


class ThreadedAsyncSandboxClient:
    """
    Mirrors AsyncSandboxClient's interface but dispatches each method call to a
    ThreadPoolExecutor where each thread maintains its own client via
    thread-local storage.
    """

    DEFAULT_MAX_WORKERS = 50

    def __init__(
        self,
        max_workers: int | None = None,
        max_connections: int = 1000,
        max_keepalive_connections: int = 200,
        **client_kwargs,
    ):
        """Initialize the threaded sandbox client."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        worker_cap = (
            max_workers if max_workers is not None else self.DEFAULT_MAX_WORKERS
        )
        self.executor = ThreadPoolExecutor(
            max_workers=worker_cap,
            thread_name_prefix="threaded-sandbox-client-executor",
        )
        self._client_thread_ids: set[int] = set()
        self._client_lock = threading.Lock()
        self.executor_name = f"threaded-sandbox-client-{id(self)}"
        register_executor(
            self.executor_name,
            self.executor,
            scaling_fn=lambda c: min(max(1, c // 8), worker_cap),
        )
        self.client_kwargs = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
            **client_kwargs,
        }
        self.logger.info(
            f"Initialized ThreadedAsyncSandboxClient (max_workers={worker_cap}, {max_connections=}, {max_keepalive_connections=})"
        )

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Dynamically proxy attribute access to dispatch method calls to the thread pool."""

        @functools.wraps(getattr(AsyncSandboxClient, name, lambda: None))
        async def wrapper(*args, **kwargs):
            def run_in_thread():
                loop = get_or_create_thread_loop()
                sandbox_client = get_or_create_thread_attr(
                    "sandbox_client",
                    AsyncSandboxClient,
                    **self.client_kwargs,
                )
                with self._client_lock:
                    self._client_thread_ids.add(threading.get_ident())
                method = getattr(sandbox_client, name)
                return loop.run_until_complete(method(*args, **kwargs))

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, run_in_thread)

        return wrapper

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int = 900,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        poll_interval: int = 3,
    ) -> Any:
        """Run a background job without occupying a worker while polling."""
        job = await self.start_background_job(
            sandbox_id,
            command,
            working_dir=working_dir,
            env=env,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = await self.get_background_job(sandbox_id, job)
            if status.completed:
                return status
            await asyncio.sleep(poll_interval)
        raise CommandTimeoutError(sandbox_id, command, timeout)

    def _close_thread_client(
        self,
        barrier: threading.Barrier | None = None,
    ) -> tuple[int, bool]:
        if barrier is not None:
            try:
                barrier.wait(timeout=5.0)
            except threading.BrokenBarrierError:
                pass

        thread_id = threading.get_ident()
        thread_local = get_thread_local_storage()
        sandbox_client = getattr(thread_local, "sandbox_client", None)
        if sandbox_client is None:
            return thread_id, False

        loop = get_or_create_thread_loop()
        try:
            loop.run_until_complete(sandbox_client.aclose())
        finally:
            delattr(thread_local, "sandbox_client")
            with self._client_lock:
                self._client_thread_ids.discard(thread_id)
        return thread_id, True

    def _close_thread_clients(self, wait: bool) -> None:
        with self._client_lock:
            pending_thread_ids = set(self._client_thread_ids)
        if not pending_thread_ids:
            return

        if not wait:
            for _ in pending_thread_ids:
                self.executor.submit(self._close_thread_client)
            return

        # Close from the owning worker thread so AsyncSandboxClient uses its
        # original event loop while releasing keepalive sockets.
        for _ in range(3):
            if not pending_thread_ids:
                break
            barrier = threading.Barrier(len(pending_thread_ids))
            futures = [
                self.executor.submit(self._close_thread_client, barrier)
                for _ in pending_thread_ids
            ]
            for future in futures:
                try:
                    thread_id, closed = future.result()
                except Exception:
                    self.logger.exception("Failed to close sandbox client in worker")
                    continue
                if closed:
                    pending_thread_ids.discard(thread_id)

        if pending_thread_ids:
            self.logger.warning(
                "Failed to close sandbox clients in %d worker thread(s)",
                len(pending_thread_ids),
            )

    def teardown(self, wait: bool = True) -> None:
        """Teardown the thread pool executor."""
        unregister_executor(self.executor_name)
        self._close_thread_clients(wait=wait)
        self.executor.shutdown(wait=wait)
