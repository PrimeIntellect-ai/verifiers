"""Sandbox lifecycle management with retry, tracking, and cleanup.

Key features:
- AsyncExitStack for guaranteed LIFO cleanup (inspired by frontier-evals)
- Centralized limits via SandboxLimits TypedDict
- Improved health checking with consecutive failure tracking
- Optional garbage collection for orphaned sandboxes
- Port allocation for exposed services
"""

import asyncio
import logging
import os
import tempfile
import time
import uuid
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from statistics import mean
from typing import Any, AsyncIterator, Protocol, Self

from prime_sandboxes import (
    AsyncSandboxClient,
    BackgroundJob as PrimeBackgroundJob,
    CommandTimeoutError as PrimeCommandTimeoutError,
    CreateSandboxRequest,
    SandboxClient,
    SandboxOOMError as PrimeSandboxOOMError,
    SandboxTimeoutError as PrimeSandboxTimeoutError,
)
from prime_sandboxes.core import APIClient

from verifiers.envs.experimental.resource_managers.errors import (
    CommandTimeoutError,
    SandboxCreationError,
    SandboxFailureInfo,
    SandboxNotReadyError,
    SandboxOOMError,
    SandboxSetupError,
    SandboxTimeoutError,
)
from verifiers.envs.experimental.resource_managers.base import (
    ManagedResource,
    ResourceManager,
    ResourceState,
)
from verifiers.envs.experimental.resource_managers.retry import RetryConfig
from verifiers.envs.experimental.resource_managers.limits import (
    DEFAULT_SANDBOX_LIMITS,
    SandboxLimits,
    merge_limits,
)
from verifiers.envs.experimental.resource_managers.recorder import (
    CommandEvent,
    NullRecorder,
    Recorder,
)
from verifiers.utils.thread_utils import get_or_create_thread_attr, get_or_create_thread_loop


logger = logging.getLogger(__name__)

class FileLock:
    """Simple file-based lock for cross-process coordination.

    Used for GC leader election - only one process becomes the GC leader
    to avoid duplicate cleanup operations.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self, blocking: bool = True, timeout: float = 0) -> bool:
        """Acquire the lock.

        Args:
            blocking: If True, wait for lock. If False, return immediately.
            timeout: Max seconds to wait (0 = no timeout, only if blocking=True)

        Returns:
            True if lock acquired, False otherwise.
        """
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT)

        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB

        start = time.monotonic()
        while True:
            try:
                fcntl.flock(self._fd, flags)
                return True
            except BlockingIOError:
                if not blocking:
                    os.close(self._fd)
                    self._fd = None
                    return False
                if timeout > 0 and (time.monotonic() - start) >= timeout:
                    os.close(self._fd)
                    self._fd = None
                    return False
                time.sleep(0.1)

    def release(self) -> None:
        """Release the lock."""
        import fcntl

        if self._fd is not None:
            with suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            self._fd = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class ThreadedAsyncSandboxClient:
    """Wraps AsyncSandboxClient to run in thread pool for better concurrency.

    Dynamically proxies all methods from AsyncSandboxClient, running them
    in a thread pool to avoid blocking the main event loop.
    """

    def __init__(
        self,
        max_workers: int = 100,
        max_connections: int = 100,
        max_keepalive_connections: int = 50,
    ):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sandbox-")
        self.client_kwargs = {
            "max_connections": max_connections,
            "max_keepalive_connections": max_keepalive_connections,
        }
        self._shutdown = False

    def __getattr__(self, name: str) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            def run() -> Any:
                loop = get_or_create_thread_loop()
                client = get_or_create_thread_attr("sandbox_client", AsyncSandboxClient, **self.client_kwargs)
                method = getattr(client, name)
                return loop.run_until_complete(method(*args, **kwargs))
            return await asyncio.get_event_loop().run_in_executor(self.executor, run)
        return wrapper

    def teardown(self) -> None:
        self._shutdown = True
        self.executor.shutdown(wait=True)


@dataclass(slots=True)
class BackgroundJob:
    """Tracks a background job running in a sandbox."""

    job_id: str
    sandbox_id: str
    command: str
    working_dir: str | None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    completed: bool = False
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    error: Exception | None = None


@dataclass
class ManagedSandbox(ManagedResource):
    """Sandbox with execution timing metrics and failure tracking."""

    request: CreateSandboxRequest | None = None
    command_times: list[float] = field(default_factory=list)
    ready_wait_time: float = 0.0
    failure_info: SandboxFailureInfo = field(default_factory=SandboxFailureInfo)
    active_jobs: dict[str, BackgroundJob] = field(default_factory=dict)
    completed_jobs: list[BackgroundJob] = field(default_factory=list)

    @property
    def avg_command_time(self) -> float:
        """Average command execution time in seconds."""
        return mean(self.command_times) if self.command_times else 0.0

    def record_failure(
        self,
        *,
        oom: bool = False,
        timeout: bool = False,
        command_timeout: bool = False,
        creation_failed: bool = False,
        setup_failed: bool = False,
        not_ready: bool = False,
        health_check_failed: bool = False,
        error_message: str | None = None,
        failed_command: str | None = None,
    ) -> None:
        """Record a failure on this sandbox."""
        if oom:
            self.failure_info.oom = True
        if timeout:
            self.failure_info.timeout = True
        if command_timeout:
            self.failure_info.command_timeout = True
        if creation_failed:
            self.failure_info.creation_failed = True
        if setup_failed:
            self.failure_info.setup_failed = True
        if not_ready:
            self.failure_info.not_ready = True
        if health_check_failed:
            self.failure_info.health_check_failed = True
        if error_message:
            self.failure_info.error_message = error_message
        if failed_command:
            self.failure_info.failed_command = failed_command
        if self.failure_info.failed_at is None:
            self.failure_info.failed_at = time.time()


class SetupCallback(Protocol):
    """Protocol for sandbox setup callbacks."""
    async def __call__(self, sandbox_id: str) -> None: ...


class SandboxManager(ResourceManager[ManagedSandbox]):
    """Manages sandbox lifecycle with proper tracking and cleanup.

    Features:
    - AsyncExitStack for guaranteed LIFO cleanup
    - Centralized limits via SandboxLimits TypedDict
    - Improved health checking with consecutive failure tracking
    - Optional garbage collection for orphaned sandboxes

    Example:
        async with sandbox_manager(default_request=request) as manager:
            sandbox = await manager.acquire(rollout_id="rollout-1")
            output = await manager.execute_command(sandbox.id, "echo hello")
        # All sandboxes released here via AsyncExitStack
    """

    # Class-level GC state (shared across all instances)
    _gc_lock_dir: Path = Path(tempfile.gettempdir()) / "verifiers-sandbox-gc"
    _gc_leader_lock: FileLock | None = None
    _gc_task: asyncio.Task[None] | None = None
    _active_managers: weakref.WeakSet["SandboxManager"] = weakref.WeakSet()

    def __init__(
        self,
        default_request: CreateSandboxRequest | None = None,
        retry_config: RetryConfig | None = None,
        limits: SandboxLimits | None = None,
        enable_gc: bool = False,
        recorder: Recorder | None = None,
        # Legacy parameters (still supported, override limits)
        timeout_per_command: int | None = None,
        wait_for_creation_max_attempts: int | None = None,
        sandbox_client_max_workers: int | None = None,
        sandbox_client_max_connections: int | None = None,
        sandbox_client_max_keepalive_connections: int | None = None,
        **kwargs: Any,
    ):
        """Initialize sandbox manager.

        Args:
            default_request: Default CreateSandboxRequest for acquire().
            retry_config: Retry configuration for transient failures.
            limits: Centralized limits (merged with DEFAULT_SANDBOX_LIMITS).
            enable_gc: Enable garbage collection for orphaned sandboxes.
            recorder: Optional recorder for tracking events (commands, state changes).
                      Use InMemoryRecorder for debugging, or implement custom Recorder.

            Legacy parameters (override limits if provided):
            timeout_per_command: Override limits["command_timeout_seconds"]
            wait_for_creation_max_attempts: Override limits["ready_max_attempts"]
            sandbox_client_max_workers: Override limits["client_max_workers"]
            sandbox_client_max_connections: Override limits["client_max_connections"]
            sandbox_client_max_keepalive_connections: Override limits["client_max_keepalive_connections"]
        """
        # Merge limits with defaults
        self.limits = merge_limits(DEFAULT_SANDBOX_LIMITS, limits)

        # Apply legacy parameter overrides
        if timeout_per_command is not None:
            self.limits["command_timeout_seconds"] = timeout_per_command
        if wait_for_creation_max_attempts is not None:
            self.limits["ready_max_attempts"] = wait_for_creation_max_attempts
        if sandbox_client_max_workers is not None:
            self.limits["client_max_workers"] = sandbox_client_max_workers
        if sandbox_client_max_connections is not None:
            self.limits["client_max_connections"] = sandbox_client_max_connections
        if sandbox_client_max_keepalive_connections is not None:
            self.limits["client_max_keepalive_connections"] = sandbox_client_max_keepalive_connections

        super().__init__(
            retry_config=retry_config,
            health_check_interval=self.limits["health_check_interval_seconds"],
            **kwargs,
        )

        self.default_request = default_request
        self.enable_gc = enable_gc

        # Convenience accessors for common limits
        self.timeout_per_command = self.limits["command_timeout_seconds"]
        self.wait_for_creation_max_attempts = self.limits["ready_max_attempts"]

        # AsyncExitStack for guaranteed cleanup
        self._exit_stack = AsyncExitStack()
        self._started = False

        # Track consecutive health check failures per sandbox
        self._health_failures: dict[str, int] = {}

        # Recorder for tracking events (commands, state changes, etc.)
        # Use NullRecorder by default (no-op) to avoid overhead when not needed
        self.recorder: Recorder = recorder or NullRecorder()

        self.client = ThreadedAsyncSandboxClient(
            max_workers=self.limits["client_max_workers"],
            max_connections=self.limits["client_max_connections"],
            max_keepalive_connections=self.limits["client_max_keepalive_connections"],
        )

        # Register this manager for GC
        SandboxManager._active_managers.add(self)

    def create_resource_object(self, rollout_id: str | None) -> ManagedSandbox:
        return ManagedSandbox(id=f"sandbox-pending-{uuid.uuid4().hex[:8]}", rollout_id=rollout_id)

    async def start(self) -> None:
        """Start the manager, setting up cleanup stack.

        Called automatically by sandbox_manager context manager.
        Can also be called manually if not using the context manager.
        """
        if self._started:
            return

        self._started = True

        # Register client teardown on exit stack (runs last)
        self._exit_stack.callback(self.client.teardown)

        # Start GC if enabled and this is the first manager
        if self.enable_gc:
            await self._maybe_start_gc()

        self.logger.debug("SandboxManager started")

    async def stop(self) -> None:
        """Stop the manager, releasing all resources via AsyncExitStack.

        Resources are released in LIFO order (last acquired = first released).
        Called automatically by sandbox_manager context manager.
        """
        if not self._started:
            return

        self.logger.info("Stopping SandboxManager...")

        # Stop health monitoring first
        await self.stop_health_monitoring()

        # Release all sandboxes
        await self.release_all()

        # Clean up exit stack (client teardown, etc.)
        await self._exit_stack.aclose()

        # Unregister from active managers
        SandboxManager._active_managers.discard(self)

        # Stop GC if no more active managers
        if not SandboxManager._active_managers and SandboxManager._gc_task:
            SandboxManager._gc_task.cancel()
            with suppress(asyncio.CancelledError):
                await SandboxManager._gc_task
            SandboxManager._gc_task = None
            if SandboxManager._gc_leader_lock:
                SandboxManager._gc_leader_lock.release()
                SandboxManager._gc_leader_lock = None

        self._started = False
        self.logger.debug("SandboxManager stopped")

    async def __aenter__(self) -> "SandboxManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit with guaranteed cleanup."""
        await self.stop()

    async def acquire(
        self,
        rollout_id: str | None = None,
        request: CreateSandboxRequest | None = None,
    ) -> ManagedSandbox:
        """Create and track a new sandbox.

        Args:
            rollout_id: Optional rollout ID for error attribution.
            request: Sandbox creation request. Uses default_request if not provided.

        Returns:
            The created ManagedSandbox.

        Raises:
            ValueError: If no request is provided and no default_request is set.
            SandboxCreationError: If sandbox creation fails.
        """
        actual_request = request or self.default_request
        if actual_request is None:
            raise ValueError("No sandbox request provided and no default_request set")

        resource = self.create_resource_object(rollout_id)
        resource.request = actual_request
        return await self._do_acquire(resource, rollout_id)

    async def create_resource(self, resource: ManagedSandbox) -> None:
        """Create sandbox via API. Retry is handled by base class."""
        if resource.request is None:
            raise ValueError("No sandbox request on resource")

        try:
            result = await self.client.create(resource.request)
            resource.id = result.id
        except Exception as e:
            resource.record_failure(creation_failed=True, error_message=str(e))
            raise SandboxCreationError(
                f"Failed to create sandbox: {e}",
                sandbox_id=resource.id,
                rollout_id=resource.rollout_id,
                cause=e,
            ) from e

    async def destroy_resource(self, resource_id: str) -> None:
        """Destroy sandbox via API. Retry is handled by base class."""
        await self.client.delete(resource_id)

    async def _check_health_impl(self, resource_id: str) -> bool:
        """Check if sandbox is healthy via echo command.

        Uses consecutive failure tracking - a sandbox is only marked unhealthy
        after `health_check_consecutive_failures` consecutive failures.
        """
        sandbox = self.get_resource(resource_id)
        timeout = self.limits["health_check_timeout_seconds"]
        max_failures = self.limits["health_check_consecutive_failures"]

        try:
            result = await self.client.execute_command(resource_id, "echo ok", timeout=timeout)
            is_healthy = result.stdout.strip() == "ok"

            if is_healthy:
                # Reset failure counter on success
                self._health_failures.pop(resource_id, None)
                return True
            else:
                # Increment failure counter
                failures = self._health_failures.get(resource_id, 0) + 1
                self._health_failures[resource_id] = failures

                if failures >= max_failures:
                    if sandbox:
                        sandbox.record_failure(
                            health_check_failed=True,
                            error_message=f"Health check failed {failures} consecutive times"
                        )
                    return False

                self.logger.warning(
                    f"Health check failed for {resource_id} ({failures}/{max_failures})"
                )
                return True  # Not yet unhealthy

        except Exception as e:
            failures = self._health_failures.get(resource_id, 0) + 1
            self._health_failures[resource_id] = failures

            if failures >= max_failures:
                if sandbox:
                    sandbox.record_failure(health_check_failed=True, error_message=str(e))
                return False

            self.logger.warning(
                f"Health check error for {resource_id} ({failures}/{max_failures}): {e}"
            )
            return True  # Not yet unhealthy

    async def wait_for_healthy(
        self,
        sandbox_id: str,
        timeout: float | None = None,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait for sandbox to become healthy with polling.

        Inspired by frontier-evals' wait_for_health pattern.
        Uses asyncio.timeout() for clean timeout handling.

        Args:
            sandbox_id: Sandbox to wait for.
            timeout: Max seconds to wait (default: ready_timeout_seconds / 2).
            poll_interval: Seconds between health checks.

        Returns:
            True if healthy, False if timeout reached or unhealthy.
        """
        if timeout is None:
            timeout = self.limits["ready_timeout_seconds"] / 2

        try:
            async with asyncio.timeout(timeout):
                while True:
                    if await self._check_health_impl(sandbox_id):
                        return True
                    await asyncio.sleep(poll_interval)
        except TimeoutError:
            self.logger.warning(f"Sandbox {sandbox_id} did not become healthy within {timeout}s")
            return False

    # ---------------------------------------------------------------------------
    # Garbage Collection (inspired by frontier-evals' Alcatraz)
    # ---------------------------------------------------------------------------

    async def _maybe_start_gc(self) -> None:
        """Try to become GC leader and start garbage collection.

        Uses file-based locking so only one process runs GC at a time.
        Non-blocking - if another process is already GC leader, we skip.
        """
        if SandboxManager._gc_task is not None:
            return  # GC already running in this process

        lock_path = SandboxManager._gc_lock_dir / "gc-leader.lock"
        lock = FileLock(lock_path)

        if lock.acquire(blocking=False):
            SandboxManager._gc_leader_lock = lock
            SandboxManager._gc_task = asyncio.create_task(self._gc_loop())
            self.logger.info("This manager is the GC leader")
        else:
            self.logger.debug("Another process is GC leader, skipping GC")

    async def _gc_loop(self) -> None:
        """Garbage collection loop - runs periodically to clean up orphans."""
        interval = self.limits["gc_interval_seconds"]

        while True:
            try:
                await asyncio.sleep(interval)
                await self._gc_cleanup_orphans()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"GC error: {e}")

    async def _gc_cleanup_orphans(self) -> None:
        """Clean up sandboxes that have been in CREATING/ERROR state too long.

        This handles cases where:
        - A process crashed without cleaning up its sandboxes
        - A sandbox got stuck in CREATING state
        - An error wasn't properly handled
        """
        threshold = self.limits["gc_orphan_threshold_seconds"]
        now = time.time()
        orphans: list[str] = []

        # Check across all active managers
        for manager in SandboxManager._active_managers:
            for sandbox in list(manager.resources.values()):
                age = now - sandbox.created_at

                # Orphan conditions:
                # 1. Stuck in CREATING for too long
                # 2. In ERROR state for too long (should have been released)
                if sandbox.state == ResourceState.CREATING and age > threshold:
                    orphans.append(sandbox.id)
                    self.logger.warning(f"GC: Orphaned sandbox {sandbox.id} (stuck creating)")
                elif sandbox.state == ResourceState.ERROR and age > threshold:
                    orphans.append(sandbox.id)
                    self.logger.warning(f"GC: Orphaned sandbox {sandbox.id} (error state)")

        if orphans:
            self.logger.info(f"GC: Cleaning up {len(orphans)} orphaned sandboxes")
            for sandbox_id in orphans:
                try:
                    # Find which manager owns this sandbox
                    for manager in SandboxManager._active_managers:
                        if sandbox_id in manager.resources:
                            await manager.release(sandbox_id)
                            break
                except Exception as e:
                    self.logger.error(f"GC: Failed to release {sandbox_id}: {e}")

    async def wait_for_ready(
        self,
        sandbox_id: str,
        setup_callback: SetupCallback | None = None,
    ) -> None:
        """Wait for sandbox to be ready, optionally run setup callback.

        Args:
            sandbox_id: The sandbox to wait for.
            setup_callback: Optional async callback to run after sandbox is ready.
                            Receives sandbox_id as argument.

        Raises:
            SandboxNotReadyError: If sandbox fails to become ready.
            SandboxSetupError: If setup callback fails.
        """
        sandbox = self.get_resource(sandbox_id)
        if sandbox is None:
            raise ValueError(f"Unknown sandbox: {sandbox_id}")

        start = time.time()
        try:
            await self._with_retry(self.client.wait_for_creation)(
                sandbox_id,
                max_attempts=self.wait_for_creation_max_attempts,
            )
        except Exception as e:
            sandbox.record_failure(not_ready=True, error_message=str(e))
            raise SandboxNotReadyError(
                f"Sandbox {sandbox_id} failed to become ready: {e}",
                sandbox_id=sandbox_id,
                rollout_id=sandbox.rollout_id,
                cause=e,
            ) from e

        sandbox.ready_wait_time = time.time() - start

        if setup_callback:
            try:
                await setup_callback(sandbox_id)
            except Exception as e:
                sandbox.record_failure(setup_failed=True, error_message=str(e))
                raise SandboxSetupError(
                    f"Sandbox {sandbox_id} setup failed: {e}",
                    sandbox_id=sandbox_id,
                    rollout_id=sandbox.rollout_id,
                    cause=e,
                ) from e

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        turn_number: int | None = None,
    ) -> str:
        """Execute command in sandbox.

        Args:
            sandbox_id: Sandbox to execute in.
            command: Command to execute.
            working_dir: Working directory for command.
            timeout: Command timeout in seconds.
            turn_number: Optional turn number for recording (agent turn tracking).

        Returns:
            Command output as string.

        Note:
            On command timeout, returns error message string rather than raising.
            OOM and sandbox timeout are recorded in failure_info and raised.
        """
        sandbox = self.get_resource(sandbox_id)
        timeout = timeout or self.timeout_per_command
        start = time.time()
        rollout_id = sandbox.rollout_id if sandbox else None

        try:
            result = await self.client.execute_command(
                sandbox_id, command, working_dir=working_dir, timeout=timeout
            )
        except PrimeCommandTimeoutError:
            duration = time.time() - start
            if sandbox:
                sandbox.command_times.append(float(timeout))
                sandbox.record_failure(command_timeout=True, failed_command=command)
            # Record timeout event
            self.recorder.record(CommandEvent.error(
                command=command,
                error_message=f"Command timed out after {timeout}s",
                duration_seconds=duration,
                resource_id=sandbox_id,
                rollout_id=rollout_id,
                turn_number=turn_number,
                working_dir=working_dir,
            ))
            return f"Error: Command timed out after {timeout}s"
        except PrimeSandboxOOMError as e:
            duration = time.time() - start
            if sandbox:
                sandbox.record_failure(oom=True, error_message=str(e), failed_command=command)
            # Record OOM event
            self.recorder.record(CommandEvent.error(
                command=command,
                error_message=f"OOM: {e}",
                duration_seconds=duration,
                resource_id=sandbox_id,
                rollout_id=rollout_id,
                turn_number=turn_number,
                working_dir=working_dir,
            ))
            raise SandboxOOMError(
                f"Sandbox {sandbox_id} ran out of memory",
                sandbox_id=sandbox_id,
                rollout_id=rollout_id,
                command=command,
                cause=e,
            ) from e
        except PrimeSandboxTimeoutError as e:
            duration = time.time() - start
            if sandbox:
                sandbox.record_failure(timeout=True, error_message=str(e), failed_command=command)
            # Record timeout event
            self.recorder.record(CommandEvent.error(
                command=command,
                error_message=f"Sandbox timeout: {e}",
                duration_seconds=duration,
                resource_id=sandbox_id,
                rollout_id=rollout_id,
                turn_number=turn_number,
                working_dir=working_dir,
            ))
            raise SandboxTimeoutError(
                f"Sandbox {sandbox_id} timed out",
                sandbox_id=sandbox_id,
                rollout_id=rollout_id,
                command=command,
                cause=e,
            ) from e

        duration = time.time() - start
        if sandbox:
            sandbox.command_times.append(duration)

        stdout = result.stdout.strip()
        stderr = (result.stderr or "").strip()

        # Record successful command
        self.recorder.record(CommandEvent.success(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=0,  # TODO: get actual exit code if available
            duration_seconds=duration,
            resource_id=sandbox_id,
            rollout_id=rollout_id,
            turn_number=turn_number,
            working_dir=working_dir,
        ))

        if stderr:
            return f"{stdout}\nstderr:\n{stderr}" if stdout else f"stderr:\n{stderr}"
        return stdout or "(no output)"

    async def start_background_job(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
    ) -> BackgroundJob:
        """Start a background job, return immediately.

        Returns:
            BackgroundJob with job_id for polling.
        """
        sandbox = self.get_resource(sandbox_id)

        try:
            result: PrimeBackgroundJob = await self._with_retry(self.client.start_background_job)(
                sandbox_id=sandbox_id,
                command=command,
                working_dir=working_dir,
            )
        except PrimeSandboxOOMError as e:
            if sandbox:
                sandbox.record_failure(oom=True, error_message=str(e), failed_command=command)
            raise SandboxOOMError(
                f"Sandbox {sandbox_id} OOM starting background job",
                sandbox_id=sandbox_id,
                rollout_id=sandbox.rollout_id if sandbox else None,
                command=command,
                cause=e,
            ) from e
        except PrimeSandboxTimeoutError as e:
            if sandbox:
                sandbox.record_failure(timeout=True, error_message=str(e), failed_command=command)
            raise SandboxTimeoutError(
                f"Sandbox {sandbox_id} timeout starting background job",
                sandbox_id=sandbox_id,
                rollout_id=sandbox.rollout_id if sandbox else None,
                command=command,
                cause=e,
            ) from e

        job = BackgroundJob(
            job_id=result.job_id,
            sandbox_id=sandbox_id,
            command=command,
            working_dir=working_dir,
        )

        if sandbox:
            sandbox.active_jobs[job.job_id] = job

        return job

    async def poll_background_job(
        self,
        sandbox_id: str,
        job: BackgroundJob,
    ) -> BackgroundJob:
        """Check job status, update job object.

        Returns:
            Updated BackgroundJob with current status.
        """
        sandbox = self.get_resource(sandbox_id)

        try:
            result = await self._with_retry(self.client.get_background_job)(
                sandbox_id, job.job_id
            )
        except PrimeSandboxOOMError as e:
            if sandbox:
                sandbox.record_failure(oom=True, error_message=str(e), failed_command=job.command)
            job.error = e
            raise SandboxOOMError(
                f"Sandbox {sandbox_id} OOM during job {job.job_id}",
                sandbox_id=sandbox_id,
                rollout_id=sandbox.rollout_id if sandbox else None,
                command=job.command,
                cause=e,
            ) from e
        except PrimeSandboxTimeoutError as e:
            if sandbox:
                sandbox.record_failure(timeout=True, error_message=str(e), failed_command=job.command)
            job.error = e
            raise SandboxTimeoutError(
                f"Sandbox {sandbox_id} timeout during job {job.job_id}",
                sandbox_id=sandbox_id,
                rollout_id=sandbox.rollout_id if sandbox else None,
                command=job.command,
                cause=e,
            ) from e

        if result.completed:
            job.completed = True
            job.completed_at = time.time()
            job.stdout = result.stdout
            job.stderr = result.stderr
            job.exit_code = result.exit_code

            if sandbox:
                sandbox.active_jobs.pop(job.job_id, None)
                sandbox.completed_jobs.append(job)

        return job

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ) -> BackgroundJob:
        """Start job, poll until complete or timeout.

        Args:
            sandbox_id: Sandbox to run in.
            command: Command to execute.
            timeout: Max seconds to wait for completion.
            working_dir: Working directory for command.
            poll_interval: Seconds between polls.

        Returns:
            Completed BackgroundJob.

        Raises:
            CommandTimeoutError: If job doesn't complete within timeout.
            SandboxOOMError: If sandbox runs out of memory.
            SandboxTimeoutError: If sandbox times out.
        """
        sandbox = self.get_resource(sandbox_id)
        job = await self.start_background_job(sandbox_id, command, working_dir)

        for elapsed in range(0, timeout + poll_interval, poll_interval):
            job = await self.poll_background_job(sandbox_id, job)
            if job.completed:
                return job

            self.logger.debug(
                f"sandbox_id={sandbox_id}: Polling job {job.job_id}... "
                f"{elapsed}/{timeout} seconds elapsed"
            )
            await asyncio.sleep(poll_interval)

        if sandbox:
            sandbox.record_failure(command_timeout=True, failed_command=command)

        raise CommandTimeoutError(
            f"Background job timed out after {timeout}s",
            sandbox_id=sandbox_id,
            rollout_id=sandbox.rollout_id if sandbox else None,
            command=command,
            timeout=timeout,
        )

    def get_failure_info(self, sandbox_id: str) -> SandboxFailureInfo | None:
        """Get failure info for a sandbox."""
        sandbox = self.get_resource(sandbox_id)
        return sandbox.failure_info if sandbox else None

    def get_failures_for_rollout(self, rollout_id: str) -> list[SandboxFailureInfo]:
        """Get all failure info for sandboxes in a rollout."""
        failures = []
        for sandbox in self.resources.values():
            if sandbox.rollout_id == rollout_id and sandbox.failure_info.has_failure:
                failures.append(sandbox.failure_info)
        return failures

    async def release_all(self) -> None:
        """Release all sandboxes using bulk delete.

        Uses itertools.batched for clean batching and bulk delete for efficiency.
        """
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.health_monitor_task
            self.health_monitor_task = None

        active_ids = [r.id for r in self.resources.values() if r.is_active]
        if not active_ids:
            return

        batch_size = self.limits["bulk_delete_batch_size"]
        self.logger.info(f"Releasing {len(active_ids)} sandboxes in batches of {batch_size}")

        try:
            sync_client = SandboxClient(APIClient())
            for batch in batched(active_ids, batch_size):
                batch_list = list(batch)  # batched yields tuples
                try:
                    sync_client.bulk_delete(sandbox_ids=batch_list)
                except Exception as e:
                    self.logger.warning(f"Bulk delete failed for batch: {e}")
                finally:
                    # Always mark as destroyed to avoid retry loops
                    for sid in batch_list:
                        if sandbox := self.resources.get(sid):
                            sandbox.mark_destroyed()
                            self._health_failures.pop(sid, None)
        except Exception as e:
            self.logger.error(f"Failed to release sandboxes: {e}")

    def teardown(self) -> None:
        """Shutdown the client thread pool."""
        self.client.teardown()

    def get_summary(self) -> dict[str, Any]:
        """Get sandbox lifecycle metrics."""
        state_counts = Counter(r.state.value for r in self.resources.values())
        error_phases = Counter(e.phase for e in self.errors)
        error_rollouts = {e.rollout_id for e in self.errors if e.rollout_id}

        # Count failure types using Counter
        failure_flags = []
        for sandbox in self.resources.values():
            info = sandbox.failure_info
            if info.oom:
                failure_flags.append("oom")
            if info.timeout:
                failure_flags.append("timeout")
            if info.command_timeout:
                failure_flags.append("command_timeout")
            if info.creation_failed:
                failure_flags.append("creation_failed")
            if info.setup_failed:
                failure_flags.append("setup_failed")
            if info.not_ready:
                failure_flags.append("not_ready")
            if info.health_check_failed:
                failure_flags.append("health_check_failed")

        failure_counts = dict(Counter(failure_flags))

        return {
            "total_sandboxes": len(self.resources),
            "state_counts": dict(state_counts),
            "total_errors": len(self.errors),
            "errors_by_phase": dict(error_phases),
            "rollouts_with_errors": len(error_rollouts),
            "failure_counts": failure_counts,
        }

    def print_summary(self, title: str = "SANDBOX SUMMARY") -> None:
        """Print sandbox lifecycle summary."""
        summary = self.get_summary()
        if summary["total_sandboxes"] == 0:
            return

        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)
        print(f"Total sandboxes: {summary['total_sandboxes']}")

        if summary["state_counts"]:
            print("\nStates:")
            for state, count in sorted(summary["state_counts"].items()):
                print(f"  {state}: {count}")

        if summary["failure_counts"]:
            print("\nFailure types:")
            for failure_type, count in sorted(summary["failure_counts"].items()):
                print(f"  {failure_type}: {count}")

        if summary["total_errors"] > 0:
            print(f"\nErrors: {summary['total_errors']}")
            for phase, count in sorted(summary["errors_by_phase"].items()):
                print(f"  {phase}: {count}")
            if summary["rollouts_with_errors"]:
                print(f"Affected rollouts: {summary['rollouts_with_errors']}")
        else:
            print("\nNo errors")

        print("=" * 60 + "\n")


@asynccontextmanager
async def sandbox_manager(
    default_request: CreateSandboxRequest | None = None,
    **kwargs: Any,
) -> AsyncIterator[SandboxManager]:
    """Context manager for sandbox lifecycle with guaranteed cleanup.

    Uses AsyncExitStack internally for LIFO cleanup order.

    Usage:
        async with sandbox_manager(default_request=request) as manager:
            sandbox = await manager.acquire(rollout_id="rollout-1")
            output = await manager.execute_command(sandbox.id, "echo hello")
        # All sandboxes released here via AsyncExitStack

    Example with limits:
        async with sandbox_manager(
            default_request=request,
            limits={"command_timeout_seconds": 60},
            enable_gc=True,
        ) as manager:
            ...

    Example with GC enabled:
        async with sandbox_manager(default_request=request, enable_gc=True) as manager:
            # This manager may become the GC leader and clean up orphaned sandboxes
            ...
    """
    manager = SandboxManager(default_request=default_request, **kwargs)
    async with manager:
        yield manager
