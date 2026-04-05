"""Sandbox environment using SandboxManager for resource lifecycle."""

import logging
from typing import Any

from prime_sandboxes import AdvancedConfigs, CreateSandboxRequest

import verifiers as vf
from verifiers.envs.experimental.resource_managers import SandboxManager

logger = logging.getLogger(__name__)


class NewSandboxEnv(vf.StatefulToolEnv):
    """Sandbox environment with managed lifecycle.

    Uses SandboxManager for:
    - Sandbox creation/deletion with retry
    - Per-rollout error tracking
    - Lifecycle observability
    - Automatic cleanup
    """

    def __init__(
        self,
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        timeout_per_command: int = 30,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        labels: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.timeout_per_command = timeout_per_command

        request = CreateSandboxRequest(
            name="sandbox",
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
            labels=labels or [],
        )

        self.sandbox_manager = SandboxManager(
            default_request=request,
            timeout_per_command=timeout_per_command,
        )
        self.add_tool(self.bash, args_to_skip=["sandbox_id"])

    async def setup_state(
        self, 
        state: vf.State, 
        **kwargs: Any
    ) -> vf.State:
        rollout_id = state.get("trajectory_id")
        sandbox = await self.sandbox_manager.acquire(rollout_id=rollout_id)
        await self.sandbox_manager.wait_for_ready(sandbox.id)
        state["sandbox_id"] = sandbox.id
        return await super().setup_state(state, **kwargs)

    async def bash(
        self, 
        command: str, 
        sandbox_id: str
    ) -> str:
        """Execute command in sandbox."""
        return await self.sandbox_manager.execute_command(
            sandbox_id, command, timeout=self.timeout_per_command
        )

    def update_tool_args(
        self, 
        tool_name: str, 
        tool_args: dict[str, Any],
        messages: vf.Messages, 
        state: vf.State, 
        **kwargs: Any
    ) -> dict[str, Any]:
        if tool_name == "bash":
            tool_args["sandbox_id"] = state["sandbox_id"]
        return tool_args

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.sandbox_manager.release(sandbox_id)

    @vf.teardown
    async def teardown(self) -> None:
        self.sandbox_manager.print_summary()
        await self.sandbox_manager.release_all()
        self.sandbox_manager.teardown()
