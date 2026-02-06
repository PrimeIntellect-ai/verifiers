import asyncio
import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import httpx
import tenacity as tc
from prime_sandboxes import (
    AsyncSandboxClient,
    CommandTimeoutError,
    CreateSandboxRequest,
    SandboxClient,
    SandboxOOMError,
    SandboxTimeoutError,
)
from prime_sandboxes.core import APIClient

import verifiers as vf
from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)

# Enable httpx debug logging if HTTPX_LOG_LEVEL is set
_httpx_log_level = os.environ.get("HTTPX_LOG_LEVEL", "").upper()
if _httpx_log_level:
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(getattr(logging, _httpx_log_level, logging.DEBUG))
    httpcore_logger = logging.getLogger("httpcore")
    httpcore_logger.setLevel(getattr(logging, _httpx_log_level, logging.DEBUG))


class SandboxCreationError(vf.SandboxError): ...


class SandboxNotReadyError(vf.SandboxError): ...


class SandboxSetupError(vf.SandboxError): ...


class ThreadedAsyncSandboxClient:
    """
    Mirrors AsyncSandboxClient's interface but dispatches each method call to a
    ThreadPoolExecutor where each thread maintains its own client via
    thread-local storage.
    """

    def __init__(
        self,
        max_workers: int = 100,
        max_connections: int = 100,
        max_keepalive_connections: int = 50,
        **client_kwargs,
    ):
        """Initialize the threaded sandbox client."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sandbox-client-executor"
        )
        self.client_kwargs = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
            **client_kwargs,
        }

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Dynamically proxy attribute access to dispatch method calls to the thread pool."""

        @functools.wraps(getattr(AsyncSandboxClient, name, lambda: None))
        async def wrapper(*args, **kwargs):
            def run_in_thread():
                loop = get_or_create_thread_loop()
                sandbox_client = get_or_create_thread_attr(
                    "sandbox_client", AsyncSandboxClient, **self.client_kwargs
                )
                method = getattr(sandbox_client, name)
                return loop.run_until_complete(method(*args, **kwargs))

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, run_in_thread)

        return wrapper

    def teardown(self, wait: bool = True) -> None:
        """Teardown the thread pool executor."""
        self.executor.shutdown(wait=wait)


class SandboxMixin:
    """Mixin providing sandbox lifecycle management with retry, tracking, and cleanup."""

    active_sandboxes: set[str]
    sandbox_client: ThreadedAsyncSandboxClient
    with_retry: Callable

    def init_sandbox_client(
        self,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
    ):
        """Initialize sandbox client and retry wrapper. Call from subclass __init__."""
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.active_sandboxes = set()
        self.sandbox_client = ThreadedAsyncSandboxClient(
            max_workers=sandbox_client_max_workers,
            max_connections=sandbox_client_max_connections,
            max_keepalive_connections=sandbox_client_max_keepalive_connections,
        )
        self.with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        ).wraps

    async def create_sandbox(self, state, request: CreateSandboxRequest) -> str:
        """Create sandbox with retry, tracking, wait_for_creation, and post-setup hook.

        Raises:
            SandboxCreationError: If sandbox creation fails after retries.
            SandboxNotReadyError: If sandbox fails to become ready.
            SandboxSetupError: If post_sandbox_setup hook fails.
        """
        try:
            sandbox = await self.with_retry(self.sandbox_client.create)(request)
        except Exception as e:
            raise SandboxCreationError(f"Failed to create sandbox: {e}") from e

        self.active_sandboxes.add(sandbox.id)
        state["sandbox_id"] = sandbox.id
        self.logger.debug(f"Created sandbox {sandbox.id}")

        try:
            await self.sandbox_client.wait_for_creation(sandbox.id)
        except Exception as e:
            raise SandboxNotReadyError(
                f"Sandbox {sandbox.id} failed to become ready: {e}"
            ) from e

        try:
            await self.post_sandbox_setup(state, self.sandbox_client)
        except vf.SandboxError:
            raise
        except Exception as e:
            raise SandboxSetupError(f"Sandbox {sandbox.id} setup failed: {e}") from e

        return sandbox.id

    async def post_sandbox_setup(
        self, state, sandbox_client: "AsyncSandboxClient | ThreadedAsyncSandboxClient"
    ):
        """Hook for subclasses to run setup after sandbox is ready."""
        pass

    async def delete_sandbox(self, sandbox_id: str):
        """Delete sandbox with retry and tracking."""

        async def _delete(sandbox_id: str):
            await self.sandbox_client.delete(sandbox_id)
            self.active_sandboxes.discard(sandbox_id)
            self.logger.debug(f"Deleted sandbox {sandbox_id}")

        try:
            await self.with_retry(_delete)(sandbox_id)
        except Exception as e:
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def bulk_delete_sandboxes(self, sandbox_ids: list[str]) -> None:
        """Delete multiple sandboxes by their IDs."""
        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(sandbox_ids)
            self.logger.debug(f"Bulk deleted sandboxes: {sandbox_ids}")
            self.active_sandboxes.difference_update(sandbox_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {sandbox_ids}: {e}")

    async def run_background_job(
        self,
        state: dict[str, Any],
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ):
        """Run a command as a background job and poll until completion or timeout."""
        sandbox_id = state["sandbox_id"]
        start_job = self.with_retry(self.sandbox_client.start_background_job)
        get_job = self.with_retry(self.sandbox_client.get_background_job)

        try:
            job = await start_job(
                sandbox_id=sandbox_id, command=command, working_dir=working_dir
            )
        except (CommandTimeoutError, httpx.ReadTimeout) as e:
            self.logger.error(f"Failed to start background job: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxOOMError as e:
            state["sandbox_oom"] = True
            self.logger.error(f"Sandbox OOM during background job: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxTimeoutError as e:
            state["sandbox_timeout"] = True
            self.logger.error(f"Sandbox timeout during background job: {repr(e)}")
            raise vf.SandboxError() from e

        try:
            for elapsed in range(0, timeout + poll_interval, poll_interval):
                results = await get_job(sandbox_id, job)
                if results.completed:
                    return results
                self.logger.debug(
                    f"{sandbox_id=}: Polling job... {elapsed} / {timeout} seconds elapsed"
                )
                await asyncio.sleep(poll_interval)
        except SandboxOOMError as e:
            state["sandbox_oom"] = True
            self.logger.error(f"Sandbox OOM during polling: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxTimeoutError as e:
            state["sandbox_timeout"] = True
            self.logger.error(f"Sandbox timeout during polling: {repr(e)}")
            raise vf.SandboxError() from e

        raise CommandTimeoutError(
            sandbox_id=sandbox_id, command=command, timeout=timeout
        )

    def teardown_sandboxes(self):
        """Delete all active sandboxes using sync client.

        Uses the synchronous SandboxClient for teardown to avoid event loop issues
        during signal handling and interpreter shutdown.
        """
        if not self.active_sandboxes:
            return
        self.logger.info(f"Deleting {len(self.active_sandboxes)} remaining sandboxes")
        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)
        batch_size = 100
        for i in range(0, len(sandbox_ids), batch_size):
            batch = sandbox_ids[i : i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sandbox_id in batch:
                    self.active_sandboxes.discard(sandbox_id)
                self.logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                self.logger.warning(f"Bulk delete failed for batch: {e}")

    def teardown_sandbox_client(self):
        """Teardown the threaded sandbox client."""
        self.sandbox_client.teardown()
