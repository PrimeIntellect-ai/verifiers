from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from datasets import Dataset
from prime_sandboxes import AdvancedConfigs

import verifiers as vf
from verifiers.types import Messages, State


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
        rubric: vf.Rubric | None = None,
        monitor_rubric: vf.Rubric | None = None,
    ):
        self.dataset = dataset
        self.task_factory = task_factory
        self.rubric = rubric
        self.monitor_rubric = monitor_rubric

    def get_dataset(self) -> Dataset:
        return self.dataset

    def get_task(self, state: State) -> Task:
        return self.task_factory(state)

    def build_rubric(self) -> vf.Rubric | None:
        return self.rubric

    def build_monitor_rubric(self) -> vf.Rubric | None:
        return self.monitor_rubric
