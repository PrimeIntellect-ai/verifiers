import asyncio
import importlib
import inspect
import types as py_types
import uuid
from collections.abc import Mapping
from typing import Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel
import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs

from .config import EnvConfig, HarnessConfig, TasksetConfig
from .harness import Harness
from .state import State
from .taskset import Taskset
from .types import ConfigData, ConfigMap, Handler
from .utils.config_utils import coerce_config, config_owner, explicit_config_data


def _package_module_name(package_id: str) -> str:
    return package_id.replace("-", "_").split("/")[-1]


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
            harness = load_harness(config.harness)
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
            taskset=explicit_config_data(self.taskset.config),
            harness=explicit_config_data(self.harness.config),
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


def load_taskset(config: TasksetConfig | ConfigData | str) -> Taskset:
    if isinstance(config, str):
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
    if isinstance(config, Mapping):
        data = dict(config)
        component_id = _optional_component_id(
            data, alias_field="taskset_id", label="taskset"
        )
        if component_id is not None:
            return cast(
                Taskset,
                _load_component(
                    component_id=component_id,
                    data=data,
                    loader_name="load_taskset",
                    base_config_cls=TasksetConfig,
                    result_cls=Taskset,
                    alias_field="taskset_id",
                    label="taskset",
                ),
            )
        return _taskset_from_config(coerce_config(TasksetConfig, data))
    if isinstance(config, TasksetConfig):
        return _taskset_from_config(config)
    raise TypeError("load_taskset expects a TasksetConfig, mapping, or id.")


def load_harness(config: HarnessConfig | ConfigData | str) -> Harness:
    if isinstance(config, str):
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
            ),
        )
    if isinstance(config, Mapping):
        data = dict(config)
        component_id = _optional_component_id(
            data, alias_field="harness_id", label="harness"
        )
        if component_id is None:
            return _harness_from_config(coerce_config(HarnessConfig, data))
        return cast(
            Harness,
            _load_component(
                component_id=component_id,
                data=data,
                loader_name="load_harness",
                base_config_cls=HarnessConfig,
                result_cls=Harness,
                alias_field="harness_id",
                label="harness",
            ),
        )
    if isinstance(config, HarnessConfig):
        return _harness_from_config(config)
    raise TypeError("load_harness expects a HarnessConfig, mapping, or id.")


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


def _load_component(
    *,
    component_id: str,
    data: ConfigData,
    loader_name: str,
    base_config_cls: type[BaseModel],
    result_cls: type[Taskset] | type[Harness],
    alias_field: str,
    label: str,
) -> Taskset | Harness:
    module = _import_component_module(component_id, label)
    loader = _component_loader(module, loader_name, component_id, label)
    config_cls = _component_config_type(
        loader=loader,
        loader_name=loader_name,
        component_id=component_id,
        base_config_cls=base_config_cls,
        label=label,
    )
    config_data = _component_config_data(
        data=data,
        component_id=component_id,
        alias_field=alias_field,
        config_cls=config_cls,
    )
    loaded = loader(config=coerce_config(config_cls, config_data))
    if not isinstance(loaded, result_cls):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} returned "
            f"{type(loaded).__name__}, expected {result_cls.__name__}."
        )
    return cast(Taskset | Harness, loaded)


def _import_component_module(component_id: str, label: str) -> object:
    module_name = _package_module_name(component_id)
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Could not import {label} package {component_id!r}. "
            f"Ensure the '{component_id}' package is installed."
        ) from exc


def _component_loader(
    module: object, loader_name: str, component_id: str, label: str
) -> Handler:
    if not hasattr(module, loader_name):
        module_name = _package_module_name(component_id)
        raise AttributeError(
            f"Module '{module_name}' does not have a '{loader_name}' function. "
            f"Install the correct {label} package or add '{loader_name}' to it."
        )
    loader = getattr(module, loader_name)
    if not callable(loader):
        raise TypeError(f"{loader_name} on {component_id!r} must be callable.")
    return cast(Handler, loader)


def _component_config_type(
    *,
    loader: Handler,
    loader_name: str,
    component_id: str,
    base_config_cls: type[BaseModel],
    label: str,
) -> type[BaseModel]:
    sig = inspect.signature(loader)
    param = sig.parameters.get("config")
    if param is None:
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must define a "
            f"required 'config' parameter annotated as a {base_config_cls.__name__} subtype."
        )
    if param.default is not inspect.Parameter.empty:
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must require "
            "'config' rather than providing a default."
        )
    if param.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must accept "
            "'config' as a normal or keyword-only parameter."
        )
    for name, extra_param in sig.parameters.items():
        if name == "config":
            continue
        if extra_param.default is inspect.Parameter.empty and extra_param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"{loader_name} for {label} package {component_id!r} must not "
                f"require parameter {name!r} in addition to 'config'."
            )
    try:
        annotation = get_type_hints(loader).get("config")
    except Exception:
        annotation = param.annotation
    if not _is_strict_component_config_type(annotation, base_config_cls):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must annotate "
            f"'config' as a {base_config_cls.__name__} subtype."
        )
    return cast(type[BaseModel], annotation)


def _is_strict_component_config_type(
    annotation: object, base_config_cls: type[BaseModel]
) -> bool:
    annotation_name = getattr(annotation, "__name__", "")
    if (
        annotation is inspect.Parameter.empty
        or annotation is object
        or annotation_name == "".join(("A", "n", "y"))
    ):
        return False
    origin = get_origin(annotation)
    if origin in (Union, py_types.UnionType) or get_args(annotation):
        return False
    return isinstance(annotation, type) and issubclass(annotation, base_config_cls)


def _component_config_data(
    *,
    data: ConfigData,
    component_id: str,
    alias_field: str,
    config_cls: type[BaseModel],
) -> ConfigData:
    config_data = dict(data)
    config_data.pop("id", None)
    config_data.pop(alias_field, None)
    if alias_field in config_cls.model_fields:
        config_data[alias_field] = component_id
    return config_data


def _optional_component_id(
    data: ConfigData, *, alias_field: str, label: str
) -> str | None:
    id_value = data.get("id")
    alias_value = data.get(alias_field)
    if id_value is not None and not isinstance(id_value, str):
        raise TypeError(f"{label}.id must be a string.")
    if alias_value is not None and not isinstance(alias_value, str):
        raise TypeError(f"{label}.{alias_field} must be a string.")
    if id_value is not None and alias_value is not None and id_value != alias_value:
        raise ValueError(
            f"{label}.id and {label}.{alias_field} must match when both are provided."
        )
    return id_value or alias_value
