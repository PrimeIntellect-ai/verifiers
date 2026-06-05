"""The environment: a taskset composed with a harness.

A ~30-line composition — not the legacy 900-line base. It exposes the tasks and
runs a rollout by delegating to the harness, then scoring the resulting transcript
with the taskset. Any taskset can be paired with any compatible harness on the fly.
"""

from pydantic_config import BaseConfig

from verifiers.nano.clients import Client
from verifiers.nano.harness import Harness, HarnessConfig, RolloutContext
from verifiers.nano.task import Task
from verifiers.nano.taskset import Taskset, TasksetConfig
from verifiers.nano.transcript import Transcript
from verifiers.nano.types import SamplingConfig


class EnvConfig(BaseConfig):
    """Two children with single field ownership. Subclass per env to narrow types."""

    taskset: TasksetConfig = TasksetConfig()
    harness: HarnessConfig = HarnessConfig()


class Environment:
    def __init__(self, taskset: Taskset, harness: Harness) -> None:
        self.taskset = taskset
        self.harness = harness

    def tasks(self) -> list[Task]:
        return self.taskset.load_tasks()

    async def run_rollout(
        self, task: Task, client: Client, model: str, sampling_args: SamplingConfig
    ) -> Transcript:
        ctx = RolloutContext(
            client=client,
            model=model,
            sampling=sampling_args,
            user=self.taskset.user,
            toolset=self.taskset.toolset,
        )
        transcript = await self.harness.rollout(
            task, ctx, transcript_cls=self.taskset.transcript_type
        )
        if transcript.error is None:
            await self.taskset.score(transcript)
        return transcript
