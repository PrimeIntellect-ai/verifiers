import math
from typing import Any, cast

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.harnesses.base import HarnessMonitorRubric
from verifiers.envs.experimental.harnesses.cli_agent import CliHarness
from verifiers.envs.experimental.task_agent_env import TaskAgentEnv
from verifiers.envs.experimental.tasksets.base import SandboxSpec, StaticTaskSet, Task
from verifiers.types import State, Tool


class CliAgentMonitorRubric(HarnessMonitorRubric):
    """Backward-compatible alias for the CLI harness monitor rubric."""


class _CliAgentTask(Task):
    def __init__(self, compat_env: "CliAgentEnv"):
        super().__init__()
        self.compat_env = compat_env

    async def get_sandbox_spec(self, state: State) -> SandboxSpec | None:
        docker_image = await self.compat_env.get_docker_image(state)
        return SandboxSpec(
            docker_image=docker_image,
            start_command=self.compat_env.start_command,
            cpu_cores=self.compat_env.cpu_cores,
            memory_gb=self.compat_env.memory_gb,
            disk_size_gb=self.compat_env.disk_size_gb,
            gpu_count=self.compat_env.gpu_count,
            timeout_minutes=max(
                1, int(math.ceil(self.compat_env.timeout_seconds / 60))
            ),
            team_id=self.compat_env.team_id,
            advanced_configs=self.compat_env.advanced_configs,
            labels=list(self.compat_env.labels or []),
        )

    async def build_env_vars(self, state: State) -> dict[str, str]:
        return await self.compat_env.build_env_vars(state)

    async def setup(self, env: Any, state: State) -> None:
        await self.compat_env.post_sandbox_setup(state)

    async def post_rollout(self, env: Any, state: State) -> None:
        await self.compat_env.post_rollout(state)


class CliAgentEnv(TaskAgentEnv):
    """
    Environment for running full agent code inside sandboxes.
    Each intercepted agent request becomes one `MultiTurnEnv` rollout step.
    """

    def __init__(
        self,
        run_command: str,
        dataset: Dataset,
        interception_port: int | None = None,
        interception_url: str | None = None,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: Any | None = None,
        labels: list[str] | None = None,
        rubric: vf.Rubric | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        sandbox_wait_for_creation_max_attempts: int = 120,
        **kwargs,
    ):
        self.run_command = run_command
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
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

        harness = CliHarness(
            run_command=run_command,
            interception_port=interception_port,
            interception_url=interception_url,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
        taskset = StaticTaskSet(
            dataset=dataset,
            task_factory=lambda state: _CliAgentTask(self),
            rubric=rubric,
        )

        super().__init__(
            harness=harness,
            taskset=taskset,
            max_turns=max_turns,
            environment_vars=None,
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            max_backoff_seconds=max_backoff_seconds,
            jitter=jitter,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
            sandbox_wait_for_creation_max_attempts=sandbox_wait_for_creation_max_attempts,
            **kwargs,
        )

        self.interception_port = harness.interception_port
        self.interception_url = harness.interception_url
        self._interception_server = harness._interception_server

    @property
    def cli_harness(self) -> CliHarness:
        return cast(CliHarness, self.harness)

    def init_interception(
        self,
        interception_port: int = 8765,
        interception_url: str | None = None,
    ) -> None:
        self.cli_harness.init_interception(interception_port, interception_url)
        self.interception_port = self.cli_harness.interception_port
        self.interception_url = self.cli_harness.interception_url
        self._interception_server = self.cli_harness._interception_server

    def _require_interception_server(self):
        return self.cli_harness._require_interception_server()

    async def get_tunnel_url(self) -> str:
        return await self.cli_harness.get_tunnel_url()

    async def get_docker_image(self, state: State) -> str:
        return self.docker_image

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = dict(self.environment_vars) if self.environment_vars else {}
        env_vars["OPENAI_BASE_URL"] = state["interception_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", "600")
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", "600")
        env_vars.setdefault("HTTPX_TIMEOUT", "600")
        model = state.get("model")
        if model:
            env_vars["OPENAI_MODEL"] = model
        return env_vars

    async def post_sandbox_setup(self, state: State) -> None:
        """Hook for post-sandbox setup. Override to upload files, run commands, etc."""

    async def check_agent_completed(self, state: State) -> bool:
        return await self.cli_harness.check_agent_completed(self, state)

    def normalize_intercepted_tools(self, intercept_tools: object) -> list[Tool] | None:
        return self.cli_harness.normalize_intercepted_tools(self, intercept_tools)

    def normalize_intercepted_messages(self, intercepted_messages: object):
        return self.cli_harness.normalize_intercepted_messages(intercepted_messages)

    async def post_rollout(self, state: State) -> None:
        """
        Override for custom post-rollout logic. For example, if sandbox state is
        needed for reward functions, compute it here before sandbox teardown.
        """
