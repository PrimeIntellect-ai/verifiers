from __future__ import annotations

from verifiers.clients import Client
from verifiers.decorators import teardown
from verifiers.envs.environment import Environment
from verifiers.rubrics.rubric import Rubric
from verifiers.types import RolloutInput, SamplingArgs, State

from .harness import Harness
from .resources import Resources
from .taskset import Taskset


class Env(Environment):
    """Experimental taskset + harness environment."""

    def __init__(
        self,
        taskset: Taskset,
        harness: Harness,
    ):
        self.resources = Resources(taskset, harness)
        attach_task_provider(self.resources.rubric, self.resources.taskset)
        super().__init__(
            dataset=self.resources.dataset,
            eval_dataset=self.resources.eval_dataset,
            parser=self.resources.rubric.parser,
            rubric=self.resources.rubric,
            env_id=self.resources.env_id,
        )

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        task = self.resources.taskset.to_task(input)
        async with self.resources.rollout(
            task,
            client=client,
            model=model,
            sampling_args=sampling_args,
        ):
            return await self.resources.harness.run(task, self.resources)

    @teardown
    async def teardown_resources(self):
        await self.resources.teardown()


def attach_task_provider(rubric: Rubric, taskset: Taskset) -> None:
    rubric.add_score_object_provider(
        lambda state: {"task": taskset.to_task(state["input"])}
    )
    rubric.add_group_score_object_provider(
        lambda states: {"tasks": [taskset.to_task(state["input"]) for state in states]}
    )
