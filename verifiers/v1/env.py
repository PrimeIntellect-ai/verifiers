import asyncio
import uuid
from typing import TypeAlias, cast

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs

from .config import EnvConfig, HarnessConfig, TasksetConfig
from .harness import Harness
from .state import State
from .taskset import Taskset
from .types import ConfigMap
from .utils.config_utils import config_owner

TasksetInput: TypeAlias = Taskset | TasksetConfig
HarnessInput: TypeAlias = Harness | HarnessConfig | None


class Env(vf.Environment):
    def __init__(
        self,
        *,
        taskset: TasksetInput | None = None,
        harness: HarnessInput = None,
        config: EnvConfig | None = None,
    ):
        if config is not None and (taskset is not None or harness is not None):
            raise TypeError("Pass either config= or taskset=/harness=, not both.")
        if config is not None:
            taskset = config.taskset
            harness = config.harness
        if taskset is None:
            raise TypeError("Env requires a taskset.")
        self.taskset = resolve_taskset(taskset)
        self.harness = resolve_harness(harness)
        self.config = EnvConfig(
            taskset=cast(TasksetConfig, self.taskset.config),
            harness=cast(HarnessConfig, self.harness.config),
        )
        self.harness.attach_taskset(self.taskset)
        super().__init__(
            dataset=self.taskset.get_dataset,
            eval_dataset=self.taskset.get_eval_dataset,
            rubric=vf.Rubric(),
        )

    @vf.teardown
    async def teardown_harness(self) -> None:
        await self.harness.teardown()

    @property
    def requires_group_rollouts(self) -> bool:
        return self.harness.runtime.has_group_stage or self._uses_custom_init_group

    @property
    def provides_advantages(self) -> bool:
        return self.harness.runtime.has_group_advantages

    @property
    def _uses_custom_init_group(self) -> bool:
        return type(self.taskset).init_group is not Taskset.init_group

    async def rollout(
        self,
        input: RolloutInput,
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        task = self.taskset.to_task(input)
        state = State.for_task(task)
        self.apply_controls(
            [state],
            {
                "client": client,
                "model": model,
                "sampling_args": sampling_args or {},
                "score_rollout": self.score_rollouts,
            },
        )
        return await self.harness.run(task, state)

    async def _run_rollout_state(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs,
    ) -> State:
        return await self.rollout(input, client, model, sampling_args)

    async def _run_group_states(
        self,
        group_inputs: list[RolloutInput],
        client: Client,
        model: str,
        sampling_args: SamplingArgs,
    ) -> list[vf.State]:
        base_task = self.taskset.to_task(group_inputs[0])
        tasks, states = await self.taskset.init_group(base_task, len(group_inputs))
        if len(tasks) != len(group_inputs) or len(states) != len(group_inputs):
            raise ValueError(
                "Taskset.init_group must return one task/state per rollout."
            )
        group_key = uuid.uuid4().hex
        for state in states:
            state.setdefault("runtime", {})
            state["runtime"]["group_key"] = group_key
        self.apply_controls(
            states,
            {
                "client": client,
                "model": model,
                "sampling_args": sampling_args,
                "score_rollout": self.score_rollouts,
            },
        )
        states = await asyncio.gather(
            *[self.harness.run(task, state) for task, state in zip(tasks, states)]
        )
        try:
            if self.score_rollouts:
                await self.harness.score_group(tasks, states)
        finally:
            await self.harness.cleanup_group(tasks, states)
        for state in states:
            state.strip_runtime_handles()
            state.assert_serializable()
        return cast(list[vf.State], states)

    def apply_controls(
        self, states: list[State], controls: ConfigMap | None = None
    ) -> list[State]:
        if controls is None:
            return states
        serializable_controls = {
            key: value for key, value in controls.items() if key != "client"
        }
        for state in states:
            state.setdefault("runtime", {})
            client = controls.get("client")
            self.harness.runtime.bind_model_client(
                state,
                cast(Client | ClientConfig | None, client)
                if client is not None
                else None,
            )
            state["runtime"].update(serializable_controls)
        return states


def resolve_taskset(value: TasksetInput) -> Taskset:
    if isinstance(value, Taskset):
        return value
    if not isinstance(value, TasksetConfig):
        raise TypeError("Env taskset must be a Taskset or TasksetConfig.")
    if type(value) is TasksetConfig:
        return Taskset(config=value)
    owner = config_owner(type(value), TasksetConfig)
    if owner is None:
        raise TypeError(
            f"No Taskset class is bound to {type(value).__name__}; "
            "instantiate the Taskset explicitly."
        )
    return cast(type[Taskset], owner)(config=value)


def resolve_harness(value: HarnessInput) -> Harness:
    if value is None:
        return Harness(config=HarnessConfig())
    if isinstance(value, Harness):
        return value
    if not isinstance(value, HarnessConfig):
        raise TypeError("Env harness must be a Harness or HarnessConfig.")
    if type(value) is HarnessConfig:
        return Harness(config=value)
    owner = config_owner(type(value), HarnessConfig)
    if owner is None:
        raise TypeError(
            f"No Harness class is bound to {type(value).__name__}; "
            "instantiate the Harness explicitly."
        )
    return cast(type[Harness], owner)(config=value)
