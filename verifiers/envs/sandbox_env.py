import asyncio
import time
import logging
from typing import Any

import tenacity as tc
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timeout_per_command_seconds = timeout_per_command_seconds
        self.sandbox_client = AsyncSandboxClient()
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
        self.add_tool(self.bash, args_to_skip=["sandbox_id"])

    async def bash(self, command: str, sandbox_id: str) -> str:
        """Execute `command` inside persistent sandbox container."""
        # sandbox_id is passed via update_tool_args, not seen by model
        s = time.time()
        await self.sandbox_client.wait_for_creation(
            sandbox_id
        )  # wait for sandbox to be created
        self.logger.debug(f"Waited {time.time() - s:.1f}s for sandbox to be ready")
        s = time.time()
        self.logger.debug(f"Executing command {command} in sandbox {sandbox_id}")
        try:
            results = await asyncio.wait_for(
                self.sandbox_client.execute_command(sandbox_id, command),
                timeout=self.timeout_per_command_seconds,
            )
        except asyncio.TimeoutError:
            e = time.time()
            timeout_msg = f"Command timed out after {self.timeout_per_command_seconds}s"
            self.logger.warning(f"{timeout_msg} in sandbox {sandbox_id}")
            return f"Error: {timeout_msg}"
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
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create per-rollout sandbox"""
        sandbox = await self.with_retry(self.sandbox_client.create)(
            self.sandbox_request
        )
        self.active_sandboxes.add(sandbox.id)
        self.logger.debug(f"Created sandbox {sandbox.id}")
        state["sandbox_id"] = sandbox.id
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        if tool_name == "bash":
            updated_args = dict(tool_args)
            updated_args["sandbox_id"] = state["sandbox_id"]
            return updated_args
        else:
            return tool_args

    async def bulk_delete_sandboxes(self, global_ids: list[str]) -> None:
        """Delete multiple sandboxes by their global IDs"""
        try:
            await self.with_retry(self.sandbox_client.bulk_delete)(
                global_ids
            )
            self.logger.debug(f"Bulk deleted sandboxes: {global_ids}")
            self.active_sandboxes.difference_update(global_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {global_ids}: {e}")

    @vf.teardown  # type: ignore
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
            batch = sandbox_ids[i:i + batch_size]
            try:
                sync_client.bulk_delete(sandbox_ids=batch)
                for sandbox_id in batch:
                    self.active_sandboxes.discard(sandbox_id)
                self.logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
            except Exception as e:
                self.logger.warning(f"Bulk delete failed for batch: {e}")
