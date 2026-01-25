import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import tenacity as tc
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest, SandboxClient
from prime_sandboxes.core import APIClient

import verifiers as vf
from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)


class SandboxCreationError(vf.SandboxError):
    """Raised when sandbox creation fails."""

    pass


class SandboxNotReadyError(vf.SandboxError):
    """Raised when sandbox fails to become ready."""

    pass


class SandboxSetupError(vf.SandboxError):
    """Raised when post-sandbox setup fails."""

    pass


logger = logging.getLogger(__name__)


class ThreadedAsyncSandboxClient:
    """Mirrors AsyncSandboxClient but dispatches to ThreadPoolExecutor with thread-local clients."""

    def __init__(
        self,
        max_workers: int = 100,
        max_connections: int = 100,
        max_keepalive_connections: int = 50,
        **client_kwargs,
    ):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sandbox-client-executor"
        )
        self.client_kwargs = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
            **client_kwargs,
        }

    def __getattr__(self, name: str) -> Callable[..., Any]:
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
        self.executor.shutdown(wait=wait)


class SandboxMixin:
    """Mixin providing sandbox lifecycle management with retry, tracking, and cleanup.

    Mirrors SandboxEnv's sandbox management so it can later be used by both
    CliAgentEnv and SandboxEnv.
    """

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
            before_sleep=tc.before_sleep_log(logger, logging.WARNING),
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
        logger.debug(f"Created sandbox {sandbox.id}")

        try:
            await self.sandbox_client.wait_for_creation(sandbox.id)
        except Exception as e:
            raise SandboxNotReadyError(
                f"Sandbox {sandbox.id} failed to become ready: {e}"
            ) from e

        try:
            await self.post_sandbox_setup(state, self.sandbox_client)
        except vf.SandboxError:
            raise  # Re-raise if already a SandboxError (from subclass override)
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

        async def _delete(sid: str):
            await self.sandbox_client.delete(sid)
            self.active_sandboxes.discard(sid)
            logger.debug(f"Deleted sandbox {sid}")

        try:
            await self.with_retry(_delete)(sandbox_id)
        except Exception as e:
            logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def bulk_delete_sandboxes(self, sandbox_ids: list[str]) -> None:
        """Delete multiple sandboxes by their IDs."""
        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(sandbox_ids)
            logger.debug(f"Bulk deleted sandboxes: {sandbox_ids}")
            self.active_sandboxes.difference_update(sandbox_ids)
        except Exception as e:
            logger.error(f"Failed to bulk delete sandboxes {sandbox_ids}: {e}")

    def teardown_sandboxes(self):
        """Bulk delete remaining sandboxes. Uses sync client for signal handling."""
        if not self.active_sandboxes:
            return
        logger.info(f"Deleting {len(self.active_sandboxes)} remaining sandboxes")
        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)
        batch_size = 100
        for i in range(0, len(sandbox_ids), batch_size):
            batch = sandbox_ids[i : i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sid in batch:
                    self.active_sandboxes.discard(sid)
                logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                logger.warning(f"Bulk delete failed for batch: {e}")

    def teardown_sandbox_client(self):
        """Teardown the threaded sandbox client."""
        self.sandbox_client.teardown()
