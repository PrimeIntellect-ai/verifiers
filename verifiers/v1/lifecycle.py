from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from verifiers.types import RolloutInput
from verifiers.utils.async_utils import maybe_retry

from .state import State
from .task import Task
from .types import ModelConfig
from .utils.json_utils import json_data

if TYPE_CHECKING:
    from .env import Env
    from .harness import Harness
    from .taskset import Taskset


class EnvRun:
    def __init__(
        self,
        *,
        env: Env | None = None,
        harness: Harness | None = None,
    ) -> None:
        if env is None and harness is None:
            raise TypeError("EnvRun requires an env or harness.")
        if env is not None and harness is not None:
            raise TypeError("EnvRun accepts env or harness, not both.")
        if harness is None:
            assert env is not None
            harness = env.harness
        self.env = env
        self.harness = harness
        self.taskset: Taskset | None = None if env is None else env.taskset
        self._entered = False

    async def __aenter__(self) -> "EnvRun":
        await self.harness.start_env_scope()
        self._entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._entered = False
        await self.harness.stop_env_scope()

    def to_task(self, input: RolloutInput | Task | str) -> Task:
        if isinstance(input, str):
            return Task(prompt=input)
        if isinstance(input, Task):
            if self.taskset is None:
                return input
            return self.taskset.to_task(input)
        if isinstance(input, dict):
            row = json_data(input)
            if self.taskset is None:
                return Task.model_validate(row)
            return self.taskset.to_task(row)
        raise TypeError("Rollout input must be a Task, string prompt, or mapping.")

    async def run_rollout(
        self,
        input: RolloutInput | Task | str,
        *,
        model: ModelConfig | str,
        teacher: ModelConfig | str | None = None,
        state: State | None = None,
        score: bool = True,
        max_retries: int = 0,
    ) -> State:
        task = self.to_task(input)
        model_config = type(self).normalize_model(model)
        teacher_config = (
            type(self).normalize_model(teacher) if teacher is not None else None
        )

        async def attempt() -> State:
            if state is None:
                rollout_state = State(task_id=task.task_id)
            elif max_retries > 0:
                rollout_state = state.model_copy(deep=True)
            else:
                rollout_state = state
            return await self.run_context(
                task,
                rollout_state,
                model=model_config,
                teacher=teacher_config,
                score=score,
            )

        return await maybe_retry(attempt, max_retries=max_retries)()

    async def run_context(
        self,
        task: Task | str,
        state: State | None = None,
        *,
        model: ModelConfig | str,
        teacher: ModelConfig | str | None = None,
        score: bool = False,
    ) -> State:
        if not self._entered:
            raise RuntimeError("EnvRun must be entered before running a context.")
        task = self.to_task(task)
        model_config = type(self).normalize_model(model)
        teacher_config = (
            type(self).normalize_model(teacher) if teacher is not None else None
        )
        state = state or State(task_id=task.task_id)
        state.task_id = task.task_id
        state.model = state.model or model_config
        state.teacher = state.teacher or teacher_config
        self.harness.initialize_extras(state)

        async with self.harness.runtime_provider_for(task).create_runtime() as runtime:
            async with AsyncExitStack() as stack:
                user = await stack.enter_async_context(
                    self.harness.rollout_user(runtime, task)
                )
                toolsets = await stack.enter_async_context(
                    self.harness.rollout_toolsets(runtime, user)
                )
                async with self.harness.open_context(
                    task=task,
                    state=state,
                    model=model_config,
                    teacher=teacher_config,
                    runtime=runtime,
                    toolsets=toolsets,
                    user=user,
                    score=score,
                ) as context:
                    if toolsets is not None:
                        toolsets.set_visibility(
                            toolsets=task.toolsets,
                            tools=task.tools,
                        )
                    await self.harness.run_lifecycle(context)
        return state

    async def group(self, rows: list[RolloutInput]) -> "Group":
        if self.env is None or self.taskset is None:
            raise RuntimeError("Grouped rollouts require an Env.")
        if not rows:
            raise ValueError("Group requires at least one row.")
        base_task = self.taskset.to_task(json_data(rows[0]))
        tasks, states = await self.taskset.init_group(base_task, len(rows))
        group_id = str(rows[0].get("example_id") or uuid.uuid4().hex)
        for state in states:
            state.group_id = state.group_id or group_id
        return Group(env_run=self, tasks=tasks, states=states)

    @staticmethod
    def normalize_model(model: ModelConfig | str) -> ModelConfig:
        if isinstance(model, str):
            return ModelConfig(model=model)
        return model


class Group:
    def __init__(
        self,
        *,
        env_run: EnvRun,
        tasks: list[Task],
        states: list[State],
    ) -> None:
        self.env_run = env_run
        self.tasks = tasks
        self.states = states

    async def run(
        self,
        *,
        model: ModelConfig | str,
        teacher: ModelConfig | str | None = None,
        max_retries: int = 0,
    ) -> list[State]:
        model_config = EnvRun.normalize_model(model)
        teacher_config = (
            EnvRun.normalize_model(teacher) if teacher is not None else None
        )
        self.states = list(
            await asyncio.gather(
                *[
                    self.env_run.run_rollout(
                        task,
                        model=model_config,
                        teacher=teacher_config,
                        state=state,
                        max_retries=max_retries,
                    )
                    for task, state in zip(self.tasks, self.states, strict=True)
                ]
            )
        )
        return await self.score(model=model_config, teacher=teacher_config)

    async def score(
        self,
        *,
        model: ModelConfig | None = None,
        teacher: ModelConfig | None = None,
    ) -> list[State]:
        if self.env_run.env is None:
            raise RuntimeError("Grouped scoring requires an Env.")
        self.states = await self.env_run.env.score_group(
            self.tasks,
            self.states,
            model=model,
            teacher=teacher,
        )
        return self.states
