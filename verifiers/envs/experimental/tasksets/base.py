from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from datasets import Dataset
from prime_sandboxes import AdvancedConfigs

import verifiers as vf
from verifiers.types import Messages, State

if TYPE_CHECKING:
    from verifiers.envs.experimental.harnesses.base import Harness


@dataclass(slots=True)
class SandboxSpec:
    docker_image: str = "python:3.11-slim"
    start_command: str = "tail -f /dev/null"
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 5
    gpu_count: int = 0
    timeout_minutes: int = 60
    environment_vars: dict[str, str] = field(default_factory=dict)
    team_id: str | None = None
    advanced_configs: AdvancedConfigs | None = None
    labels: list[str] = field(default_factory=list)


class Task:
    def __init__(self, sandbox: SandboxSpec | None = None):
        self.sandbox = sandbox

    async def get_sandbox_spec(self, state: State) -> SandboxSpec | None:
        return self.sandbox

    async def build_env_vars(self, state: State) -> dict[str, str]:
        return {}

    async def setup(self, env: Any, state: State) -> None:
        pass

    async def prompt(self, state: State) -> Messages | str | None:
        return state.get("prompt")

    async def post_rollout(self, env: Any, state: State) -> None:
        pass

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        """Optional environment/user response hook for task-driven multi-turn flows."""
        return []

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return None


class TaskSet:
    def get_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_task(self, state: State) -> Task:
        raise NotImplementedError

    def get_task_config(self) -> dict[str, Any] | None:
        return None

    def get_agent_workdir(self) -> str | None:
        return None

    def build_harness(self) -> "Harness | None":
        from verifiers.envs.experimental.harnesses.config import (
            build_harness_from_config,
        )

        return build_harness_from_config(
            self.get_task_config(),
            agent_workdir=self.get_agent_workdir(),
        )

    def build_rubric(self) -> vf.Rubric | None:
        return None

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return None


class StaticTaskSet(TaskSet):
    """TaskSet backed by a fixed dataset and task factory."""

    def __init__(
        self,
        dataset: Dataset,
        task_factory: Callable[[State], Task],
        task_config: dict[str, Any] | None = None,
        agent_workdir: str | None = None,
        rubric: vf.Rubric | None = None,
        monitor_rubric: vf.Rubric | None = None,
    ):
        self.dataset = dataset
        self.task_factory = task_factory
        self.task_config = dict(task_config) if task_config is not None else None
        self.agent_workdir = agent_workdir
        self.rubric = rubric
        self.monitor_rubric = monitor_rubric

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_task(self, state: State) -> Task:
        return self.task_factory(state)

    def get_task_config(self) -> dict[str, Any] | None:
        return self.task_config

    def get_agent_workdir(self) -> str | None:
        return self.agent_workdir

    def build_rubric(self) -> vf.Rubric | None:
        return self.rubric

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return self.monitor_rubric
