import asyncio
import uuid
from typing import cast

from pydantic import BaseModel
import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs

from .config import (
    EnvConfig,
    HarnessConfig,
    TasksetConfig,
)
from .harness import Harness
from .state import State
from .taskset import Taskset
from .types import ConfigData, ConfigMap
from .utils.component_utils import (
    call_component_loader,
    component_config_data,
    component_config_type,
    component_loader,
    import_component_module,
)
from .utils.config_utils import coerce_config, config_owner


class Env(vf.Environment):
    def __init__(
        self,
        *,
        taskset: Taskset | None = None,
        harness: Harness | None = None,
        config: EnvConfig | None = None,
    ):
        if config is not None and (taskset is not None or harness is not None):
            raise TypeError("Pass either config= or taskset=/harness=, not both.")
        if config is not None:
            taskset = load_taskset(config.taskset)
            harness = load_harness(config.harness, taskset=taskset)
        if taskset is None:
            raise TypeError("Env requires a taskset.")
        if not isinstance(taskset, Taskset):
            raise TypeError("Env taskset must be a Taskset object.")
        if harness is None:
            harness = Harness(config=HarnessConfig())
        elif not isinstance(harness, Harness):
            raise TypeError("Env harness must be a Harness object.")
        self.taskset = taskset
        self.harness = harness
        self.config = EnvConfig(
            taskset=self.taskset.config,
            harness=self.harness.config,
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


def load_taskset(config: TasksetConfig | str) -> Taskset:
    if isinstance(config, str):
        if not config:
            raise ValueError("taskset.id must be a non-empty string.")
        return cast(
            Taskset,
            _load_component(
                component_id=config,
                data={},
                loader_name="load_taskset",
                base_config_cls=TasksetConfig,
                result_cls=Taskset,
                alias_field="taskset_id",
                label="taskset",
            ),
        )
    if isinstance(config, TasksetConfig):
        if config.taskset_id == "":
            raise ValueError("taskset.taskset_id must be a non-empty string.")
        loaded = _load_component_from_config(
            config=config,
            component_id=config.taskset_id,
            loader_name="load_taskset",
            base_config_cls=TasksetConfig,
            result_cls=Taskset,
            label="taskset",
        )
        if loaded is not None:
            return cast(Taskset, loaded)
        return _taskset_from_config(config)
    raise TypeError("load_taskset expects a TasksetConfig or id.")


def load_harness(
    config: HarnessConfig | str, *, taskset: Taskset | None = None
) -> Harness:
    if isinstance(config, str):
        if not config:
            raise ValueError("harness.id must be a non-empty string.")
        return cast(
            Harness,
            _load_component(
                component_id=config,
                data={},
                loader_name="load_harness",
                base_config_cls=HarnessConfig,
                result_cls=Harness,
                alias_field="harness_id",
                label="harness",
                taskset=taskset,
            ),
        )
    if isinstance(config, HarnessConfig):
        if config.harness_id == "":
            raise ValueError("harness.harness_id must be a non-empty string.")
        loaded = _load_component_from_config(
            config=config,
            component_id=config.harness_id,
            loader_name="load_harness",
            base_config_cls=HarnessConfig,
            result_cls=Harness,
            label="harness",
            taskset=taskset,
        )
        if loaded is not None:
            return cast(Harness, loaded)
        return _harness_from_config(config)
    raise TypeError("load_harness expects a HarnessConfig or id.")


def _taskset_from_config(config: TasksetConfig) -> Taskset:
    if type(config) is TasksetConfig:
        return Taskset(config=config)
    owner = config_owner(type(config), TasksetConfig)
    if owner is None:
        raise TypeError(
            f"No Taskset class is bound to {type(config).__name__}; "
            "instantiate the Taskset explicitly."
        )
    return cast(type[Taskset], owner)(config=config)


def _harness_from_config(config: HarnessConfig) -> Harness:
    if type(config) is HarnessConfig:
        return Harness(config=config)
    owner = config_owner(type(config), HarnessConfig)
    if owner is None:
        raise TypeError(
            f"No Harness class is bound to {type(config).__name__}; "
            "instantiate the Harness explicitly."
        )
    return cast(type[Harness], owner)(config=config)


def _load_component_from_config(
    *,
    config: BaseModel,
    component_id: str | None,
    loader_name: str,
    base_config_cls: type[BaseModel],
    result_cls: type[Taskset] | type[Harness],
    label: str,
    taskset: Taskset | None = None,
) -> Taskset | Harness | None:
    if component_id is None:
        return None
    try:
        module = import_component_module(component_id, label)
        loader = component_loader(module, loader_name, component_id, label)
    except (AttributeError, ValueError):
        if config_owner(type(config), base_config_cls) is not None:
            return None
        raise
    config_cls = component_config_type(
        loader=loader,
        loader_name=loader_name,
        component_id=component_id,
        base_config_cls=base_config_cls,
        label=label,
    )
    if not isinstance(config, config_cls):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} expects "
            f"{config_cls.__name__}, got {type(config).__name__}."
        )
    loaded = call_component_loader(loader, config=config, taskset=taskset)
    if not isinstance(loaded, result_cls):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} returned "
            f"{type(loaded).__name__}, expected {result_cls.__name__}."
        )
    return cast(Taskset | Harness, loaded)


def _load_component(
    *,
    component_id: str,
    data: ConfigData,
    loader_name: str,
    base_config_cls: type[BaseModel],
    result_cls: type[Taskset] | type[Harness],
    alias_field: str,
    label: str,
    taskset: Taskset | None = None,
) -> Taskset | Harness:
    module = import_component_module(component_id, label)
    loader = component_loader(module, loader_name, component_id, label)
    config_cls = component_config_type(
        loader=loader,
        loader_name=loader_name,
        component_id=component_id,
        base_config_cls=base_config_cls,
        label=label,
    )
    config_data = component_config_data(
        data=data,
        component_id=component_id,
        alias_field=alias_field,
        config_cls=config_cls,
    )
    loaded = call_component_loader(
        loader, config=coerce_config(config_cls, config_data), taskset=taskset
    )
    if not isinstance(loaded, result_cls):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} returned "
            f"{type(loaded).__name__}, expected {result_cls.__name__}."
        )
    return cast(Taskset | Harness, loaded)
