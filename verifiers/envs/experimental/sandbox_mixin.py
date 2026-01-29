import asyncio
import functools
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import httpx
import tenacity as tc
from prime_sandboxes import (
    AsyncSandboxClient,
    CommandTimeoutError,
    CreateSandboxRequest,
    DownloadTimeoutError,
    SandboxOOMError,
    SandboxClient,
    SandboxNotRunningError,
    SandboxTimeoutError,
    SandboxUnresponsiveError,
    UploadTimeoutError,
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
    import httpx  # noqa: F401 - import ensures httpx is loaded before logger config

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(getattr(logging, _httpx_log_level, logging.DEBUG))
    # Also enable httpcore for lower-level connection debugging
    httpcore_logger = logging.getLogger("httpcore")
    httpcore_logger.setLevel(getattr(logging, _httpx_log_level, logging.DEBUG))


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

    # Watchdog configuration
    SLOW_CALL_THRESHOLD_SECS = 60  # Log stack trace after this many seconds
    WATCHDOG_INTERVAL_SECS = 30  # How often watchdog checks for slow calls

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

        # Track in-flight calls for watchdog: call_id -> {start_time, thread_id, method, arg0}
        self._inflight_calls: dict[str, dict[str, Any]] = {}
        self._inflight_lock = threading.Lock()
        self._logged_slow_calls: set[str] = set()  # Avoid repeated logging

        # Start watchdog thread
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="sandbox-client-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Background thread that monitors for slow/stuck calls and logs stack traces."""
        while not self._watchdog_stop.wait(self.WATCHDOG_INTERVAL_SECS):
            self._check_slow_calls()

    def _check_slow_calls(self) -> None:
        """Check for calls exceeding threshold and log their stack traces."""
        now = time.time()
        with self._inflight_lock:
            inflight_snapshot = dict(self._inflight_calls)

        for call_id, call_info in inflight_snapshot.items():
            elapsed = now - call_info["start_time"]
            if elapsed >= self.SLOW_CALL_THRESHOLD_SECS:
                # Only log once per call (at threshold crossing)
                if call_id in self._logged_slow_calls:
                    continue
                self._logged_slow_calls.add(call_id)

                thread_id = call_info.get("thread_id")
                method = call_info.get("method", "?")
                arg0 = call_info.get("arg0", "")

                # Get stack trace for the thread if available
                stack_trace = "Stack trace unavailable"
                if thread_id and thread_id in sys._current_frames():
                    frame = sys._current_frames()[thread_id]
                    stack_lines = traceback.format_stack(frame)
                    stack_trace = "".join(stack_lines)

                logger.warning(
                    f"[SANDBOX_CALL_SLOW] {call_id} method={method} arg0={arg0} "
                    f"blocked for {elapsed:.0f}s\n{stack_trace}"
                )

    def __getattr__(self, name: str) -> Callable[..., Any]:
        @functools.wraps(getattr(AsyncSandboxClient, name, lambda: None))
        async def wrapper(*args, **kwargs):
            call_id = f"{name}_{threading.current_thread().name}_{time.time():.0f}"
            # Log call start with first arg (usually sandbox_id) for context
            first_arg = args[0] if args else ""
            logger.debug(
                f"[SANDBOX_CALL_START] {call_id} method={name} arg0={first_arg}"
            )
            start_time = time.time()

            # Track this call for watchdog (thread_id updated once thread starts)
            call_info = {
                "start_time": start_time,
                "thread_id": None,
                "method": name,
                "arg0": first_arg,
            }
            with self._inflight_lock:
                self._inflight_calls[call_id] = call_info

            def run_in_thread():
                current_thread = threading.current_thread()
                thread_name = current_thread.name
                # Update thread_id for watchdog stack trace capture
                with self._inflight_lock:
                    if call_id in self._inflight_calls:
                        self._inflight_calls[call_id]["thread_id"] = (
                            current_thread.ident
                        )
                logger.debug(f"[SANDBOX_THREAD_START] {call_id} thread={thread_name}")
                try:
                    loop = get_or_create_thread_loop()
                    sandbox_client = get_or_create_thread_attr(
                        "sandbox_client", AsyncSandboxClient, **self.client_kwargs
                    )
                    method = getattr(sandbox_client, name)
                    result = loop.run_until_complete(method(*args, **kwargs))
                    logger.debug(f"[SANDBOX_THREAD_END] {call_id} success=True")
                    return result
                except (
                    SandboxNotRunningError,
                    CommandTimeoutError,
                    UploadTimeoutError,
                    DownloadTimeoutError,
                ) as e:
                    logger.debug(
                        f"[SANDBOX_THREAD_END] {call_id} success=False "
                        f"error={type(e).__name__}: {e}"
                    )
                    raise vf.SandboxError(f"{type(e).__name__}: {e}") from e
                except Exception as e:
                    logger.debug(
                        f"[SANDBOX_THREAD_END] {call_id} success=False "
                        f"error={type(e).__name__}: {e}"
                    )
                    raise

            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, run_in_thread)
                duration = time.time() - start_time
                logger.debug(
                    f"[SANDBOX_CALL_END] {call_id} duration={duration:.1f}s success=True"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.debug(
                    f"[SANDBOX_CALL_END] {call_id} duration={duration:.1f}s "
                    f"success=False error={type(e).__name__}"
                )
                raise
            finally:
                # Remove from tracking
                with self._inflight_lock:
                    self._inflight_calls.pop(call_id, None)
                    self._logged_slow_calls.discard(call_id)

        return wrapper

    def teardown(self, wait: bool = True) -> None:
        # Stop watchdog thread
        self._watchdog_stop.set()
        if self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)
        self.executor.shutdown(wait=wait)

    def log_thread_pool_state(self) -> dict[str, int]:
        """Log and return thread pool state for debugging hangs.

        Returns dict with thread pool metrics for optional caller use.
        """
        executor = self.executor
        # ThreadPoolExecutor internals (Python 3.8+)
        active_threads = len(executor._threads)
        pending_tasks = executor._work_queue.qsize()
        max_workers = executor._max_workers

        logger.info(
            f"[THREAD_POOL] active_threads={active_threads}/{max_workers} "
            f"pending_tasks={pending_tasks}"
        )

        return {
            "active_threads": active_threads,
            "pending_tasks": pending_tasks,
            "max_workers": max_workers,
        }


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

    async def run_background_job(
        self,
        state: dict[str, Any],
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
        sandbox_client: "AsyncSandboxClient | ThreadedAsyncSandboxClient | None" = None,
        start_retry: Callable | None = None,
        poll_retry: Callable | None = None,
    ):
        """Run a command as a background job and poll until completion or timeout."""
        sandbox_id = state["sandbox_id"]
        client = sandbox_client or self.sandbox_client
        start_job = (
            start_retry(client.start_background_job)
            if start_retry
            else client.start_background_job
        )
        get_job = (
            poll_retry(client.get_background_job)
            if poll_retry
            else client.get_background_job
        )

        try:
            job = await start_job(
                sandbox_id=sandbox_id, command=command, working_dir=working_dir
            )
        except (CommandTimeoutError, httpx.ReadTimeout) as e:
            logger.error(f"Failed to start background job: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxUnresponsiveError as e:
            state["sandbox_unresponsive"] = True
            logger.error(f"Background job failed: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxOOMError as e:
            state["sandbox_oom"] = True
            logger.error(f"Sandbox OOM during background job: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxTimeoutError as e:
            state["sandbox_timeout"] = True
            logger.error(f"Sandbox timeout during background job: {repr(e)}")
            raise vf.SandboxError() from e

        try:
            for elapsed in range(0, timeout + poll_interval, poll_interval):
                results = await get_job(sandbox_id, job)
                if results.completed:
                    return results
                logger.debug(
                    f"{sandbox_id=}: Polling job... {elapsed} / {timeout} seconds elapsed"
                )
                await asyncio.sleep(poll_interval)
        except SandboxUnresponsiveError as e:
            state["sandbox_unresponsive"] = True
            logger.error(f"Sandbox unresponsive during polling: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxOOMError as e:
            state["sandbox_oom"] = True
            logger.error(f"Sandbox OOM during polling: {repr(e)}")
            raise vf.SandboxError() from e
        except SandboxTimeoutError as e:
            state["sandbox_timeout"] = True
            logger.error(f"Sandbox timeout during polling: {repr(e)}")
            raise vf.SandboxError() from e

        raise CommandTimeoutError(
            sandbox_id=sandbox_id, command=command, timeout=timeout
        )

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
