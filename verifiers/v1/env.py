import asyncio
import uuid
from collections.abc import Callable
from typing import TypeAlias, cast

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs

from .harness import Harness
from .state import State
from .taskset import Taskset
from .types import ConfigMap
from .config import EnvConfig, HarnessConfig, TasksetConfig

TasksetBuilder: TypeAlias = type[Taskset] | Callable[..., Taskset]
HarnessBuilder: TypeAlias = type[Harness] | Callable[..., Harness]


class Env(vf.Environment):
    def __init__(
        self,
        taskset: Taskset,
        harness: Harness | None = None,
    ):
        harness = Harness() if harness is None else harness
        self.taskset = taskset
        self.harness = harness
        self.harness.attach_taskset(taskset)
        super().__init__(
            dataset=self.taskset.get_dataset,
            eval_dataset=self.taskset.get_eval_dataset,
            rubric=vf.Rubric(),
        )

    @classmethod
    def config(
        cls,
        *,
        taskset: TasksetBuilder = Taskset,
        harness: HarnessBuilder = Harness,
        taskset_config: type[TasksetConfig] | None = None,
        harness_config: type[HarnessConfig] | None = None,
    ) -> type[EnvConfig]:
        taskset_config = taskset_config or getattr(
            taskset, "_config_cls", TasksetConfig
        )
        harness_config = harness_config or getattr(
            harness, "_config_cls", HarnessConfig
        )
        taskset_name = str(getattr(taskset, "__name__", type(taskset).__name__))
        harness_name = str(getattr(harness, "__name__", type(harness).__name__))
        name = (
            f"{taskset_name}EnvConfig"
            if harness is Harness
            else f"{taskset_name}{harness_name}EnvConfig"
        )
        return type(
            name,
            (EnvConfig,),
            {
                "__module__": cls.__module__,
                "__annotations__": {
                    "taskset": taskset_config,
                    "harness": harness_config,
                },
                "taskset": taskset_config(),
                "harness": harness_config(),
            },
        )

    @classmethod
    def loader(
        cls,
        *,
        taskset: TasksetBuilder = Taskset,
        harness: HarnessBuilder = Harness,
        taskset_config: type[TasksetConfig] | None = None,
        harness_config: type[HarnessConfig] | None = None,
        env_config: type[EnvConfig] | None = None,
    ):
        env_config_cls = env_config or cls.config(
            taskset=taskset,
            harness=harness,
            taskset_config=taskset_config,
            harness_config=harness_config,
        )

        def load_environment(config=None) -> "Env":
            return cls.from_config(
                config,
                taskset=taskset,
                harness=harness,
                env_config=env_config_cls,
            )

        load_environment.__annotations__["config"] = env_config_cls
        return load_environment

    @classmethod
    def from_config(
        cls,
        config: EnvConfig | None = None,
        *,
        taskset: TasksetBuilder = Taskset,
        harness: HarnessBuilder = Harness,
        env_config: type[EnvConfig] | None = None,
    ) -> "Env":
        if env_config is not None:
            config_cls = env_config
        elif isinstance(config, EnvConfig):
            config_cls = type(config)
        else:
            config_cls = cls.config(
                taskset=taskset,
                harness=harness,
            )
        env_config_value = config_cls.from_config(config)
        return cls(
            taskset=taskset(config=env_config_value.taskset),
            harness=harness(config=env_config_value.harness),
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
