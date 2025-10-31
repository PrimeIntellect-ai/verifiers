import atexit
import signal
import time
from typing import Any

import verifiers as vf

try:
    from prime_sandboxes import (
        AdvancedConfigs,
        APIClient,
        AsyncSandboxClient,
        CreateSandboxRequest,
        SandboxClient,
    )
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
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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

        # Install handlers for regular exception, sigint (Ctrl-C) and sigterm (standard termination signal)
        atexit.register(self.cleanup_sandboxes)
        signal.signal(
            signal.SIGINT,
            lambda sig, frame: (
                self.cleanup_sandboxes(),
                signal.default_int_handler(sig, frame),
            ),
        )
        signal.signal(
            signal.SIGTERM, lambda _, __: (self.cleanup_sandboxes(), exit(143))
        )

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
        results = await self.sandbox_client.execute_command(sandbox_id, command)
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

    async def post_rollout(self, messages: vf.Messages, state: vf.State, **kwargs):
        """
        Override for custom post-rollout logic. For example, if sandbox state is needed for reward functions,
        run computation here and cache the result in state before sandbox is destroyed.
        """
        pass

    async def destroy_sandbox(self, sandbox_id: str | None) -> None:
        if sandbox_id is None:
            return
        try:
            await self.sandbox_client.delete(sandbox_id)
            self.active_sandboxes.discard(sandbox_id)
            self.logger.debug(f"Deleted sandbox {sandbox_id}")
        except Exception as e:
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create per-rollout sandbox"""
        sandbox = await self.sandbox_client.create(self.sandbox_request)
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

    async def is_completed(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> bool:
        """
        When overriding, if sandbox state is needed for reward functions,
        run computation here and cache the result in state.
        """
        completed = await super().is_completed(messages, state, **kwargs)
        if completed:
            await self.post_rollout(messages, state, **kwargs)
            await self.destroy_sandbox(state.pop("sandbox_id"))
        return completed

    def bulk_delete_sandboxes(self, global_ids: list[str]) -> None:
        """Delete multiple sandboxes by their global IDs"""
        sandbox_client = SandboxClient(APIClient())
        try:
            sandbox_client.bulk_delete(global_ids)
            self.logger.debug(f"Bulk deleted sandboxes: {global_ids}")
            self.active_sandboxes.difference_update(global_ids)
        except Exception as e:
            self.logger.error(f"Failed to bulk delete sandboxes {global_ids}: {e}")

    def cleanup_sandboxes(self):
        """Delete all active sandboxes"""
        if len(self.active_sandboxes) == 0:
            return
        self.logger.info(
            f"Cleaning up {len(self.active_sandboxes)} remaining sandboxes"
        )
        sandbox_client = SandboxClient(APIClient())
        # TODO: Use the SandboxClient.bulk_delete method once it is more stable and faster
        while self.active_sandboxes:
            successfully_deleted = set()
            for sandbox_id in self.active_sandboxes:
                try:
                    self.logger.debug(f"Deleting sandbox {sandbox_id}")
                    sandbox_client.delete(sandbox_id)
                    successfully_deleted.add(sandbox_id)
                    self.logger.debug(f"Successfully deleted sandbox {sandbox_id}")
                except Exception as e:
                    self.logger.error(f"Failed to delete sandbox {sandbox_id}: {e}")

            self.active_sandboxes -= successfully_deleted

            # If no sandboxes were deleted in this pass, break to avoid infinite loop
            if not successfully_deleted:
                self.logger.error(
                    f"Unable to delete remaining sandboxes: {self.active_sandboxes}"
                )
                break
