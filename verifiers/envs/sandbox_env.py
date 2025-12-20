import asyncio
import functools
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from verifiers.utils.thread_utils import (
    get_or_create_thread_attr,
    get_or_create_thread_loop,
)

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


import tenacity as tc
from prime_sandboxes import CommandTimeoutError

import verifiers as vf

try:
    from prime_sandboxes import (
        AdvancedConfigs,
        AsyncSandboxClient,
        CreateSandboxRequest,
        SandboxClient,
    )
    from prime_sandboxes.core import APIClient
except ImportError:
    raise ImportError(
        "prime-sandboxes is not installed. Please install it with `uv pip install prime-sandboxes`."
    )


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
            max_workers=max_workers,
            thread_name_prefix="sandbox-client-executor",
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
                    "sandbox_client",
                    AsyncSandboxClient,
                    **self.client_kwargs,
                )
                method = getattr(sandbox_client, name)
                return loop.run_until_complete(method(*args, **kwargs))

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, run_in_thread)

        return wrapper

    def teardown(self, wait: bool = True) -> None:
        """Teardown the thread pool executor."""
        self.executor.shutdown(wait=wait)


class SandboxState(TypedDict):
    ready: bool


class SandboxCreationError(vf.SandboxError): ...


class SandboxNotReadyError(vf.SandboxError): ...


class SandboxEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        sandbox_name: str = "sandbox-env",
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        timeout_per_command_seconds: int = 30,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        stop_errors: list[type[Exception]] | None = None,
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        **kwargs,
    ):
        super().__init__(
            stop_errors=stop_errors if stop_errors is not None else [vf.SandboxError],
            **kwargs,
        )
        self.timeout_per_command_seconds = timeout_per_command_seconds
        self.sandbox_client = ThreadedAsyncSandboxClient(
            max_workers=sandbox_client_max_workers,
            max_connections=sandbox_client_max_connections,
            max_keepalive_connections=sandbox_client_max_keepalive_connections,
        )
        self.sandbox_request = CreateSandboxRequest(
            name=sandbox_name,
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_minutes=timeout_minutes,
            environment_vars=environment_vars,
            team_id=team_id,
            advanced_configs=advanced_configs,
        )
        self.active_sandboxes = set()
        self.with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.ERROR),
            reraise=True,
        ).wraps
        self.add_tool(
            self.bash, args_to_skip=["sandbox_id", "working_dir", "sandbox_state"]
        )

    async def _wait_for_sandbox_ready(
        self, sandbox_state: SandboxState, sandbox_id: str
    ):
        """Wait for sandbox to be created"""
        s = time.time()
        self.logger.debug(f"Waiting for sandbox {sandbox_id} to be ready")
        try:
            await self.sandbox_client.wait_for_creation(sandbox_id)
            sandbox_state["ready"] = True
        except Exception as e:
            raise SandboxNotReadyError(e)
        self.logger.debug(f"Waited {time.time() - s:.1f}s for sandbox to be ready")

    async def bash(
        self,
        command: str,
        sandbox_id: str,
        sandbox_state: SandboxState,
        working_dir: str | None = None,
    ) -> str:
        """Execute `command` inside persistent sandbox container."""
        # sandbox_id is passed via update_tool_args, not seen by model
        if not sandbox_state["ready"]:
            await self._wait_for_sandbox_ready(sandbox_state, sandbox_id)

        s = time.time()
        self.logger.debug(f"Executing command {command} in sandbox {sandbox_id}")
        try:
            results = await self.sandbox_client.execute_command(
                sandbox_id,
                command,
                working_dir=working_dir,
                timeout=self.timeout_per_command_seconds,
            )
        except CommandTimeoutError:
            e = time.time()
            timeout_msg = f"Command timed out after {self.timeout_per_command_seconds}s"
            self.logger.warning(f"{timeout_msg} in sandbox {sandbox_id}")
            return f"Error: {timeout_msg}"
        except Exception as e:
            raise vf.SandboxError(cause=e)
        e = time.time()
        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        combined = stdout
        if stderr:
            if combined:
                combined = f"{combined}\nstderr:\n{stderr}"
            else:
                combined = f"stderr:\n{stderr}"
        output = combined or "(no output)"
        self.logger.debug(f"Executed command in {e - s:.1f}s. Got output: {output}")
        return output

    async def post_rollout(self, state: vf.State):
        """
        Override for custom post-rollout logic. For example, if sandbox state is needed for reward functions,
        run computation here and cache the result in state before sandbox is destroyed.
        """
        pass

    @vf.cleanup
    async def destroy_sandbox(self, state: vf.State):
        await self.post_rollout(state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id is None:
            return

        async def _delete_sandbox(sandbox_id: str):
            await self.sandbox_client.delete(sandbox_id)
            self.active_sandboxes.discard(sandbox_id)
            self.logger.debug(f"Deleted sandbox {sandbox_id}")

        try:
            await self.with_retry(_delete_sandbox)(sandbox_id)
        except Exception as e:
            # only warn, not raise an error on deletion
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create per-rollout sandbox"""
        try:
            sandbox = await self.with_retry(self.sandbox_client.create)(
                self.sandbox_request
            )
        except Exception as e:
            raise SandboxCreationError(e)
        self.active_sandboxes.add(sandbox.id)
        self.logger.debug(f"Created sandbox {sandbox.id}")
        state["sandbox_id"] = sandbox.id
        state["sandbox_state"] = {"ready": False}
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        updated_args = dict(tool_args)
        if tool_name == "bash":
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["sandbox_state"] = state["sandbox_state"]
            updated_args["working_dir"] = state.get("working_dir")
        return updated_args

    async def bulk_delete_sandboxes(self, global_ids: list[str]) -> None:
        """Delete multiple sandboxes by their global IDs"""
        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(global_ids)
            self.logger.debug(f"Bulk deleted sandboxes: {global_ids}")
            self.active_sandboxes.difference_update(global_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {global_ids}: {e}")

    @vf.teardown
    async def teardown_sandboxes(self):
        """Delete all active sandboxes using sync client.

        Uses the synchronous SandboxClient for teardown to avoid event loop issues
        during signal handling and interpreter shutdown.
        """
        if len(self.active_sandboxes) == 0:
            return
        self.logger.info(f"Deleting {len(self.active_sandboxes)} remaining sandboxes")

        # Use sync client for teardown - avoids event loop issues during shutdown
        sync_client = SandboxClient(APIClient())
        sandbox_ids = list(self.active_sandboxes)

        # Delete in batches of 100
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

    @vf.teardown
    async def teardown_sandbox_client(self):
        self.sandbox_client.teardown()
