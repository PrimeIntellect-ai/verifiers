from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast, final

from verifiers.types import (
    RolloutInput,
)
from verifiers.utils.async_utils import maybe_retry

from .config import Config
from .harness import Harness, HarnessConfig
from .runtime import (
    RuntimeConfig,
    RuntimeConfigValue,
    RuntimeProvider,
)
from .state import State
from .task import Task
from .taskset import Taskset, TasksetConfig, collect_owner_signals
from .types import Handler, JsonData, ModelClient, ModelConfig
from .utils.scoring_utils import score_group as score_group_signals

if TYPE_CHECKING:
    from datasets import Dataset


@final
class EnvConfig(Config):
    taskset: TasksetConfig = TasksetConfig()
    harness: HarnessConfig = HarnessConfig()
    runtime: RuntimeConfig | None = None


class Env:
    def __init__(
        self,
        *,
        taskset: Taskset,
        harness: Harness | None = None,
        runtime: RuntimeProvider | RuntimeConfigValue | None = None,
    ):
        if not isinstance(taskset, Taskset):
            raise TypeError("Env taskset must be a Taskset.")
        if harness is not None and not isinstance(harness, Harness):
            raise TypeError("Env harness must be a Harness.")
        self.taskset = taskset
        self.harness = harness or Harness()
        self.harness.bind(taskset=self.taskset, runtime=runtime)
        self.runtime_config = self.harness.runtime_config
        self.runtime_provider = self.harness.runtime_provider
        self.config = EnvConfig(
            taskset=cast(TasksetConfig, self.taskset.config),
            harness=cast(HarnessConfig, self.harness.config),
            runtime=self.runtime_config,
        )
        self.env_id = ""
        self.env_args: JsonData = {}
        self.pass_threshold = 0.5

    @property
    def requires_group_rollouts(self) -> bool:
        uses_custom_init_group = type(self.taskset).init_group is not Taskset.init_group
        return (
            self.taskset.has_group_signals
            or any(signal["stage"] == "group" for signal in self.harness.signals)
            or uses_custom_init_group
        )

    @property
    def provides_advantages(self) -> bool:
        return self.taskset.has_advantages or any(
            signal["kind"] == "advantage" for signal in self.harness.signals
        )

    def get_dataset(self, n: int = -1, seed: int | None = None) -> "Dataset":
        dataset = self.taskset.get_dataset()
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            return dataset.select(range(min(n, len(dataset))))
        return dataset

    def get_eval_dataset(self, n: int = -1, seed: int | None = None) -> "Dataset":
        dataset = self.taskset.get_eval_dataset()
        if not len(dataset):
            dataset = self.taskset.get_dataset()
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            return dataset.select(range(min(n, len(dataset))))
        return dataset

    async def run_handlers_for_group(
        self,
        kind: str,
        tasks: list[Task],
        states: list[State],
        teacher: ModelClient | None = None,
    ) -> None:
        handlers = [
            *cast(list[Handler], self.taskset.handlers[kind]),
            *cast(list[Handler], self.harness.handlers[kind]),
        ]
        for handler in handlers:
            if getattr(handler, f"{kind}_stage", "rollout") != "group":
                continue
            result = await self.harness.call_handler(
                handler,
                tasks[0],
                states[0],
                tasks=tasks,
                states=states,
                teacher=teacher,
                teacher_client=teacher.client if teacher is not None else None,
                teacher_model=teacher.config.model if teacher is not None else None,
            )
            if result is not None:
                raise TypeError(f"Group {kind} handlers must mutate states in place.")

    async def run_rollout(
        self,
        input: RolloutInput | Task,
        *,
        model: ModelConfig,
        teacher: ModelConfig | None = None,
        state: State | None = None,
        max_retries: int = 0,
    ) -> State:
        task = self.taskset.to_task(
            input if isinstance(input, Task) else cast(JsonData, input)
        )

        async def attempt() -> State:
            rollout_state = state or State(task_id=task.task_id)
            rollout_state.task_id = task.task_id
            rollout_state.model = model
            rollout_state.teacher = teacher
            return await self.harness.run(
                task,
                rollout_state,
                model=model,
                teacher=teacher,
                score=True,
            )

        return await maybe_retry(attempt, max_retries=max_retries)()

    async def score_group(
        self,
        tasks: list[Task],
        states: list[State],
        *,
        model: ModelConfig | None = None,
        teacher: ModelConfig | None = None,
    ) -> list[State]:
        model_config = (
            model if model is not None else (states[0].model if states else None)
        )
        teacher_config = (
            teacher if teacher is not None else (states[0].teacher if states else None)
        )
        if model_config is not None:
            for state in states:
                if state.model is None:
                    state.model = model_config
                elif state.model != model_config:
                    raise ValueError("Group states must use one model config.")
        if teacher_config is not None:
            for state in states:
                if state.teacher is None:
                    state.teacher = teacher_config
                elif state.teacher != teacher_config:
                    raise ValueError("Group states must use one teacher config.")
        group_id = uuid.uuid4().hex
        for state in states:
            state.group_id = state.group_id or group_id
        model_client = (
            self.harness.load_model_client(model_config)
            if model_config is not None
            else None
        )
        teacher_client = (
            self.harness.load_model_client(teacher_config)
            if teacher_config is not None
            else None
        )
        try:
            await score_group_signals(
                collect_owner_signals(self.taskset, self.harness),
                tasks,
                states,
                model_client=model_client,
                teacher=teacher_client,
            )
            await self.run_handlers_for_group(
                "update", tasks, states, teacher=teacher_client
            )
        finally:
            await self.run_handlers_for_group(
                "cleanup", tasks, states, teacher=teacher_client
            )
            if teacher_client is not None:
                await self.harness.close_model_client(teacher_client)
            if model_client is not None:
                await self.harness.close_model_client(model_client)
            for state in states:
                self.harness.validate_extras(state)
                state.finalize()
        return states

    async def close(self) -> None:
        await self.harness.close()
