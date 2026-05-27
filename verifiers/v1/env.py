import asyncio
import uuid
from typing import TypeAlias, cast

from pydantic import ValidationInfo, field_validator
import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs

from .config import Config
from .harness import Harness, HarnessConfig
from .state import State
from .taskset import Taskset, TasksetConfig
from .types import ConfigMap
from .utils.config_utils import explicit_config_data

TasksetInput: TypeAlias = Taskset
HarnessInput: TypeAlias = Harness | None


class EnvConfig(Config):
    taskset: TasksetConfig = TasksetConfig()
    harness: HarnessConfig = HarnessConfig()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        extra_fields = set(cls.model_fields) - set(EnvConfig.model_fields)
        if extra_fields:
            raise TypeError(
                f"{cls.__name__} defines unsupported root env config fields: "
                f"{', '.join(sorted(extra_fields))}. Put env-specific settings on "
                "a TasksetConfig or HarnessConfig instead."
            )
        for field_name, expected_type in (
            ("taskset", TasksetConfig),
            ("harness", HarnessConfig),
        ):
            annotation = cls.model_fields[field_name].annotation
            if not (
                isinstance(annotation, type) and issubclass(annotation, expected_type)
            ):
                raise TypeError(
                    f"{cls.__name__}.{field_name} must be typed as a "
                    f"{expected_type.__name__} subclass."
                )

    @field_validator("taskset", "harness", mode="before")
    @classmethod
    def validate_child_config(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            raise ValueError(
                f"EnvConfig.{info.field_name} cannot be None. "
                "Omit the section to use the default config."
            )
        try:
            explicit_config_data(value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return value


class Env(vf.Environment):
    def __init__(
        self,
        *,
        taskset: TasksetInput | None = None,
        harness: HarnessInput = None,
    ):
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
    raise TypeError("Env taskset must be a Taskset.")


def resolve_harness(value: HarnessInput) -> Harness:
    if value is None:
        return Harness(config=HarnessConfig())
    if isinstance(value, Harness):
        return value
    raise TypeError("Env harness must be a Harness.")
