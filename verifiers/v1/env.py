import asyncio
import inspect
import uuid
from collections.abc import Callable
from types import UnionType
from typing import TypeAlias, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel
import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs

from .config import ConfigSource, EnvConfig, HarnessConfig, TasksetConfig
from .harness import Harness
from .state import State
from .taskset import Taskset
from .types import ConfigMap
from .utils.config_utils import explicit_config_data

TasksetBuilder: TypeAlias = type[Taskset] | Callable[..., Taskset] | Taskset
HarnessBuilder: TypeAlias = type[Harness] | Callable[..., Harness] | Harness
ConfigT = TypeVar("ConfigT", bound=BaseModel)


class Env(vf.Environment):
    def __init__(
        self,
        config: ConfigSource,
        *,
        taskset: TasksetBuilder = Taskset,
        harness: HarnessBuilder = Harness,
    ):
        taskset_config_cls = builder_config_type(taskset, TasksetConfig)
        harness_config_cls = builder_config_type(harness, HarnessConfig)
        if isinstance(config, EnvConfig):
            taskset_input = config.taskset
            harness_input = config.harness
        else:
            data = explicit_config_data(config)
            extra_keys = set(data) - set(EnvConfig.model_fields)
            if extra_keys:
                raise ValueError(f"Unknown Env config keys: {sorted(extra_keys)}.")
            null_sections = [
                key
                for key in ("taskset", "harness")
                if key in data and data[key] is None
            ]
            if null_sections:
                raise ValueError(
                    f"Env config sections cannot be null: {sorted(null_sections)}."
                )
            taskset_input = data.get("taskset")
            harness_input = data.get("harness")
        if not isinstance(taskset, Taskset):
            taskset = taskset(
                config=taskset_config_cls.model_validate(
                    explicit_config_data(taskset_input)
                )
            )
            if inspect.isawaitable(taskset):
                if inspect.iscoroutine(taskset):
                    taskset.close()
                raise TypeError("Env taskset builders must be synchronous.")
        if not isinstance(harness, Harness):
            harness = harness(
                config=harness_config_cls.model_validate(
                    explicit_config_data(harness_input)
                )
            )
            if inspect.isawaitable(harness):
                if inspect.iscoroutine(harness):
                    harness.close()
                raise TypeError("Env harness builders must be synchronous.")
        self.taskset = taskset
        self.harness = harness
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


def builder_config_type(
    builder: TasksetBuilder | HarnessBuilder, base: type[ConfigT]
) -> type[ConfigT]:
    if isinstance(builder, Taskset | Harness):
        return cast(type[ConfigT], type(builder.config))
    if inspect.iscoroutinefunction(builder):
        raise TypeError("Env builder callables must be synchronous.")
    config_cls = getattr(builder, "_config_cls", None)
    if isinstance(config_cls, type) and issubclass(config_cls, base):
        return cast(type[ConfigT], config_cls)
    signature = inspect.signature(cast(Callable[..., Taskset | Harness], builder))
    if "config" not in signature.parameters:
        raise TypeError("Env builder callables must accept a config parameter.")
    try:
        annotation = get_type_hints(builder).get("config")
    except Exception:
        annotation = signature.parameters["config"].annotation
    config_cls = config_type(annotation, base)
    if config_cls is None:
        raise TypeError(
            "Env builder config parameters must be annotated with a config type."
        )
    return config_cls


def config_type(annotation: object, base: type[ConfigT]) -> type[ConfigT] | None:
    if annotation is inspect.Parameter.empty:
        return None
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for arg in get_args(annotation):
            config_cls = config_type(arg, base)
            if config_cls is not None:
                return config_cls
        return None
    if isinstance(annotation, type) and issubclass(annotation, base):
        return annotation
    return None
