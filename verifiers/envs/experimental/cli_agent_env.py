import asyncio
import logging
import math
import os
import time
from typing import Any

from prime_sandboxes import (
    AdvancedConfigs,
    BackgroundJob,
    BackgroundJobStatus,
    CreateSandboxRequest,
)

import verifiers as vf
from verifiers.envs.experimental.api_env import ApiEnv
from verifiers.envs.experimental.sandbox_mixin import (
    SandboxMixin,
    SandboxMonitorRubric,
    SandboxTimeouts,
)
from verifiers.types import State

logger = logging.getLogger(__name__)


class AgentError(vf.InfraError):
    """Raised when the agent process fails or exits unexpectedly."""


def make_agent_error(state: State, message: str) -> AgentError:
    """Create an AgentError with rollout-specific sandbox context when available."""
    context_parts = [
        f"sandbox_id={state['sandbox_id']}",
        f"rollout_id={state['rollout_id']}",
        f"example_id={state['example_id']}",
    ]
    state_info = state["input"].get("info", {})
    instance_id = state_info.get("instance_id")
    if instance_id:
        context_parts.append(f"instance_id={instance_id}")
    return AgentError(f"{message} ({', '.join(context_parts)})")


class CliAgentMonitorRubric(vf.Rubric):
    """Monitor rubric that tracks CLI agent execution state."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.agent_error)

    async def agent_error(self, state: vf.State) -> float:
        """Whether the agent errored (non-zero exit_code)."""
        agent_exit_code = state.get("agent_exit_code")
        if agent_exit_code is None:
            return 0.0
        return float(agent_exit_code != 0)


class CliAgentEnv(SandboxMixin, ApiEnv):
    """
    Environment for running full agent code inside sandboxes.
    Extends ApiEnv with sandbox lifecycle management. The agent runs as a
    CLI command inside a remote sandbox, and its API requests are intercepted
    via the same HTTP proxy server used by ApiEnv.
    """

    def __init__(
        self,
        run_command: str,
        interception_port: int | None = None,
        interception_url: str | None = None,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 5.0,
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        labels: list[str] | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 50,
        sandbox_client_max_connections: int = 1000,
        sandbox_client_max_keepalive_connections: int = 200,
        sandbox_wait_for_creation_max_attempts: int = 120,
        sandbox_creations_per_minute: float | None = 128,
        timeouts: SandboxTimeouts = SandboxTimeouts(),
        keep_sandbox_for_scoring: bool = False,
        **kwargs,
    ):
        super().__init__(
            interception_port=interception_port,
            interception_url=interception_url,
            use_tunnel=True,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            **kwargs,
        )
        self.init_sandbox_client(
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            max_backoff_seconds=max_backoff_seconds,
            jitter=jitter,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
            sandbox_wait_for_creation_max_attempts=sandbox_wait_for_creation_max_attempts,
            sandbox_creations_per_minute=sandbox_creations_per_minute,
            timeouts=timeouts,
        )
        self.keep_sandbox_for_scoring = keep_sandbox_for_scoring
        self.run_command = run_command
        self.docker_image = docker_image
        self.start_command = start_command
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.environment_vars = environment_vars
        self.team_id = team_id
        self.advanced_configs = advanced_configs
        self.labels = labels

        self.add_rubric(SandboxMonitorRubric())
        self.add_rubric(CliAgentMonitorRubric())

    async def launch_agent(self, state: State) -> None:
        """Create sandbox and start the agent command as a background job."""
        env_vars = await self.build_env_vars(state)
        docker_image = await self.get_docker_image(state)
        resources = self.get_sandbox_resources(state)

        rollout_id = state["rollout_id"]
        sandbox_request = CreateSandboxRequest(
            name=rollout_id,
            docker_image=docker_image,
            start_command=self.start_command,
            cpu_cores=resources["cpu_cores"],
            memory_gb=resources["memory_gb"],
            disk_size_gb=resources["disk_size_gb"],
            gpu_count=resources["gpu_count"],
            gpu_type=resources.get("gpu_type"),
            vm=resources.get("vm", resources["gpu_count"] > 0),
            timeout_minutes=resources["timeout_minutes"],
            environment_vars=env_vars,
            team_id=self.team_id,
            advanced_configs=self.advanced_configs,
            labels=self.labels if self.labels else [],
        )
        self.logger.debug(
            f"Creating sandbox with OPENAI_BASE_URL={env_vars.get('OPENAI_BASE_URL')} "
            f"docker_image={docker_image}"
        )
        await self.create_sandbox(state, sandbox_request)
        await self.start_agent(state)

    async def cleanup_agent(self, state: State) -> None:
        """Cancel completion wait task and clean up background job."""
        task = state.get("completion_wait_task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        state.pop("background_job", None)

    async def get_docker_image(self, state: State) -> str:
        """Get the Docker image for the sandbox. Override for per-task images."""
        return self.docker_image

    def get_sandbox_resources(self, state: State) -> dict[str, Any]:
        """Get sandbox resource allocation. Override for per-instance resources."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_size_gb": self.disk_size_gb,
            "gpu_count": self.gpu_count,
            "gpu_type": None,
            "vm": self.gpu_count > 0,
            "timeout_minutes": math.ceil(self.timeout_seconds / 60),
        }

    PROTECTED_ENV_VARS = frozenset(
        {
            "OPENAI_BASE_URL",
            "OPENAI_TIMEOUT",
            "OPENAI_REQUEST_TIMEOUT",
            "HTTPX_TIMEOUT",
            "OPENAI_MODEL",
            "OPENAI_API_KEY",
        }
    )

    async def build_env_vars(self, state: State) -> dict[str, str]:
        """Build environment variables for the sandbox. Override to add custom vars."""
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "3600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "3600")
        env_vars.setdefault("HTTPX_TIMEOUT", "3600")
        secret = os.environ.get("INTERCEPTION_SECRET")
        if secret:
            env_vars["OPENAI_API_KEY"] = secret
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Hook for post-sandbox setup. Override to upload files, run commands, etc."""
        pass

    async def start_agent(self, state: State) -> None:
        """Start the agent command using background job."""
        sandbox_id = state["sandbox_id"]

        self.logger.debug(f"Starting agent in sandbox {sandbox_id}")
        try:
            background_job: BackgroundJob = (
                await self.sandbox_client.start_background_job(
                    sandbox_id,
                    self.run_command,
                )
            )
        except Exception as e:
            raise vf.SandboxError(f"Failed to start agent: {e}") from e
        state["background_job"] = background_job
        state["agent_start_time"] = time.time()

        state["completion_wait_task"] = asyncio.create_task(
            self.wait_for_completion(state)
        )

    async def wait_for_completion(self, state: State) -> None:
        """Poll for agent completion using background job API."""
        sandbox_id = state.get("sandbox_id")
        background_job: BackgroundJob | None = state.get("background_job")

        if not sandbox_id or not background_job:
            state["agent_completed"] = True
            return

        try:
            await asyncio.wait_for(
                self.poll_job_completion(state, sandbox_id, background_job),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Agent timed out after {self.timeout_seconds}s")
            state["agent_timed_out"] = True
            state["error"] = make_agent_error(
                state, f"Agent timed out after {self.timeout_seconds}s"
            )
        except asyncio.CancelledError:
            self.logger.debug("Completion wait task cancelled")
            raise
        except Exception as e:
            error = make_agent_error(state, f"Agent polling failed: {e}")
            state["error"] = error
            self.logger.error(str(error))
        finally:
            state["agent_completed"] = True

    async def poll_job_completion(
        self, state: State, sandbox_id: str, background_job: BackgroundJob
    ) -> None:
        """Poll until background job completes, capturing output."""
        while True:
            status: BackgroundJobStatus = await self.sandbox_client.get_background_job(
                sandbox_id, background_job, timeout=self.timeouts.poll
            )
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                if status.exit_code == 0:
                    self.logger.debug(
                        f"Agent completed successfully (exit_code={status.exit_code})"
                    )
                else:
                    stderr_full = status.stderr or ""
                    num_turns = len(state.get("trajectory", []))
                    if num_turns == 0:
                        error = make_agent_error(
                            state,
                            f"Agent crashed before any LLM call "
                            f"(exit_code={status.exit_code}): {stderr_full}",
                        )
                    else:
                        error = make_agent_error(
                            state,
                            f"Agent crashed after {num_turns} turn(s) "
                            f"(exit_code={status.exit_code}): {stderr_full}",
                        )
                    state["error"] = error
                    self.logger.error(str(error))
                return
            await asyncio.sleep(self.poll_interval)

    @vf.cleanup
    async def destroy_sandbox(self, state: State):
        """Cleanup sandbox after rollout.

        When `keep_sandbox_for_scoring` is True, sandbox deletion is deferred
        (e.g. when the rubric needs sandbox access during scoring).
        The sandbox is still deregistered from active tracking so the
        environment teardown does not attempt a redundant bulk-delete.

        If the rollout was not completed (e.g. cancelled during shutdown),
        the sandbox is always deleted since scoring will not happen.
        """
        completed = state.get("is_completed", False)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            if self.keep_sandbox_for_scoring and completed:
                self.deregister_sandbox(sandbox_id)
            else:
                await self.delete_sandbox(sandbox_id)
