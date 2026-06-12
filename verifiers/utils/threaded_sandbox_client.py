import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from prime_sandboxes import (
    AsyncSandboxClient,
    CommandTimeoutError,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxTimeoutError,
)

from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
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
    READINESS_FAILURE_ATTEMPTS = 5

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
                loop = get_or_create_thread_attr("loop", asyncio.new_event_loop)
                asyncio.set_event_loop(loop)
                sandbox_client = get_or_create_thread_attr(
                    "sandbox_client",
                    AsyncSandboxClient,
                    **self.client_kwargs,
                )
                method = getattr(sandbox_client, name)
                return loop.run_until_complete(method(*args, **kwargs))

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, run_in_thread)

        return wrapper

    async def wait_for_creation_resilient(
        self,
        sandbox_id: str,
        max_attempts: int = 60,
        stability_checks: int = 1,
    ) -> None:
        """Wait for readiness without occupying one worker for the entire poll loop."""
        consecutive_successes = 0
        readiness_failures = 0
        last_readiness_error: Exception | None = None
        for attempt in range(max_attempts):
            sandbox = await self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                try:
                    await self.execute_command(
                        sandbox_id,
                        "echo 'sandbox ready'",
                        timeout=10,
                    )
                except Exception as exc:
                    consecutive_successes = 0
                    readiness_failures += 1
                    last_readiness_error = exc
                    if readiness_failures >= self.READINESS_FAILURE_ATTEMPTS:
                        raise SandboxNotRunningError(
                            sandbox_id,
                            status="RUNNING",
                            message=(
                                f"Sandbox {sandbox_id} stayed unreachable after "
                                f"{readiness_failures} readiness checks: {exc}"
                            ),
                        ) from exc
                else:
                    readiness_failures = 0
                    consecutive_successes += 1
                    if consecutive_successes >= stability_checks:
                        return
                    await asyncio.sleep(0.5)
                    continue
            elif sandbox.status in {"ERROR", "TERMINATED", "TIMEOUT"}:
                error_type = sandbox.error_type
                error_class = {
                    "OOM_KILLED": SandboxOOMError,
                    "TIMEOUT": SandboxTimeoutError,
                    "IMAGE_PULL_FAILED": SandboxImagePullError,
                }.get(error_type, SandboxNotRunningError)
                message = (
                    f"Sandbox {sandbox_id} failed ({error_type}): "
                    f"{sandbox.error_message}"
                    if sandbox.error_message
                    else None
                )
                raise error_class(
                    sandbox_id,
                    status=sandbox.status,
                    error_type=error_type,
                    message=message,
                )

            await asyncio.sleep(1 if attempt < 5 else 2)

        message = "Timeout during sandbox creation"
        if last_readiness_error is not None:
            message += f": {last_readiness_error}"
        raise SandboxNotRunningError(sandbox_id, status=message)

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

    def teardown(self, wait: bool = True) -> None:
        """Teardown the thread pool executor."""
        unregister_executor(self.executor_name)
        self.executor.shutdown(wait=wait)
