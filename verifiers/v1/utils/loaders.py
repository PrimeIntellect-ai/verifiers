"""Resolve taskset, harness, judge, hook, and environment plugins."""

import functools
import importlib
import importlib.util
import pkgutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import ValidationError
from pydantic_config import BaseConfig

from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.configs.task import DecoratedFunctionConfig, RewardFunctionConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.env import Env
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.harness import Harness
from verifiers.v1.judge import Judge, judge_config_cls
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset
from verifiers.v1.utils.generic import concrete_type, prefix_validation_error


def builtin_harness_ids() -> list[str]:
    """The harness ids that ship with verifiers (the `verifiers.v1.harnesses`
    subpackages)."""
    from verifiers.v1 import harnesses

    return sorted(m.name for m in pkgutil.iter_modules(harnesses.__path__))


def narrow_plugin_field(
    data: dict,
    field: str,
    resolve: Callable[[str], type],
    default_id: str | None = None,
) -> None:
    raw = data.get(field)
    if isinstance(raw, BaseConfig):
        raw = raw.model_dump()
    raw = dict(raw or {})
    ident = raw.get("id") or default_id
    if not ident:
        return
    if not isinstance(ident, str):
        # A dangling `--<field>.id` (no value) parses as boolean True.
        hint = (
            f"; the built-in harnesses are: {', '.join(builtin_harness_ids())}"
            if field == "harness"
            else ""
        )
        raise ValueError(  # noqa: TRY004 - Pydantic validators must raise ValueError
            f"{field}.id needs an id, and none was given (got {ident!r}); "
            f"pass the id right after the flag{hint}"
        )
    try:
        data[field] = resolve(ident).model_validate({**raw, "id": ident})
    except ValidationError as e:
        # Validated here, inside the owner's mode="before" validator, the errors
        # would surface without their `<field>` segment — the CLI would render a
        # flag path missing what the user typed.
        raise prefix_validation_error(e, (field,)) from None


def _import_plugin(plugin_id: str, kind: str, group: str) -> ModuleType:
    # Hub ids are `owner/name[@version]` (installed as just `name`); strip both
    # before normalizing so a hub id imports the same module a bare id would.
    name = plugin_id.rsplit("/", 1)[-1].split("@", 1)[0]
    module = name.replace("-", "_").lower()
    namespaced = f"{group}.{module}"
    target = namespaced if importlib.util.find_spec(namespaced) else module
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        if kind == "harness" and plugin_id == "default":
            raise ModuleNotFoundError(
                "the `default` harness was renamed to `bash`: "
                "--env.agent.harness.id bash (or your env's role name)"
            ) from e
        hint = (
            f" The built-in harnesses are: {', '.join(builtin_harness_ids())}."
            if kind == "harness"
            else ""
        )
        article = "An" if kind[0] in "aeiou" else "A"
        raise ModuleNotFoundError(
            f"{kind} {plugin_id!r} not found (tried to import {target!r}). {article} {kind} is a "
            f"package exporting its {kind.capitalize()} subclass via `__all__` — the built-in "
            f"ones ship with verifiers in the `{group}` package; any other package "
            f"must already be installed.{hint}"
        ) from e


def _plugin_class(module: ModuleType, base: type, kind: str) -> type:
    names = getattr(module, "__all__", None)
    if names is None:
        raise AttributeError(
            f"{kind} module {module.__name__!r} defines no `__all__`; a {kind} must export its "
            f"{base.__name__} subclass via `__all__` (e.g. `__all__ = ['My{base.__name__}']`)."
        )
    matches = [
        obj
        for name in names
        if isinstance(obj := getattr(module, name, None), type)
        and issubclass(obj, base)
        and obj is not base
    ]
    if not matches:
        raise TypeError(
            f"{kind} module {module.__name__!r} exports no {base.__name__} subclass via "
            f"`__all__` (found {list(names)}); export exactly one."
        )
    if len(matches) > 1:
        # ValueError, not TypeError: "no plugin here" (TypeError) is a state the
        # taskset-fallback callers legitimately swallow — an ambiguous export is an
        # authoring error that must stay loud everywhere.
        raise ValueError(
            f"{kind} module {module.__name__!r} exports {len(matches)} {base.__name__} "
            f"subclasses via `__all__` ({[c.__name__ for c in matches]}); export exactly one."
        )
    return matches[0]


def import_taskset(taskset_id: str) -> ModuleType:
    return _import_plugin(taskset_id, "taskset", "verifiers.v1.tasksets")


def import_harness(harness_id: str) -> ModuleType:
    return _import_plugin(harness_id, "harness", "verifiers.v1.harnesses")


def import_judge(judge_id: str) -> ModuleType:
    return _import_plugin(judge_id, "judge", "verifiers.v1.judges")


def taskset_class(taskset_id: str) -> type[Taskset]:
    return _plugin_class(import_taskset(taskset_id), Taskset, "taskset")


def harness_class(harness_id: str) -> type[Harness]:
    return _plugin_class(import_harness(harness_id), Harness, "harness")


def judge_class(judge_id: str) -> type[Judge]:
    return _plugin_class(import_judge(judge_id), Judge, "judge")


def default_harness_id(taskset_id: str) -> str:
    if not taskset_id:
        return "bash"
    try:
        module = import_taskset(taskset_id)
        _plugin_class(module, Harness, "harness")
    except (ModuleNotFoundError, TypeError, AttributeError):
        return "bash"
    return taskset_id


def import_environment(env_id: str) -> ModuleType:
    return _import_plugin(env_id, "environment", "verifiers.v1.envs")


def environment_class(taskset_id: str, env_id: str = "") -> type[Env]:
    """The `Env` class for a run. An explicit `env_id` names it directly,
    and a failure to resolve raises — an explicit pairing must not silently fall
    back. Otherwise the taskset's own: its package's exported `Env`
    subclass, else `SingleAgentEnv`."""
    if env_id:
        return _plugin_class(import_environment(env_id), Env, "environment")
    if not taskset_id:
        return SingleAgentEnv
    try:
        module = import_taskset(taskset_id)
        return _plugin_class(module, Env, "environment")
    except (ModuleNotFoundError, TypeError, AttributeError):
        return SingleAgentEnv


def load_environment(config: EnvConfig) -> Env:
    """Construct the env for `config`. Every construction site (eval, serve, gepa)
    goes through here so subclass envs load everywhere."""
    return environment_class(config.taskset.id, config.id)(config)


def load_taskset(config: TasksetConfig) -> Taskset:
    return taskset_class(config.id)(config)


def load_harness(config: HarnessConfig) -> Harness:
    return harness_class(config.id)(config)


def load_judge(config: JudgeConfig) -> Judge:
    return judge_class(config.id)(config)


@functools.cache
def file_module(path: str) -> ModuleType:
    """Import a standalone `.py` file, cached by path so repeated plugged-fn loads
    share one module object."""
    resolved = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(resolved.stem, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plugged_fn(path: str) -> Callable[..., Any]:
    """Resolve a plugged function's import path: `pkg.module.function`,
    `pkg.module:function`, or `path/to/file.py:function`."""
    module_path, _, attr = path.rpartition(":")
    if not module_path:
        module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"fn {path!r} must be `pkg.module.function`, `pkg.module:function`, "
            "or `path/to/file.py:function`"
        )
    if module_path.endswith(".py"):
        module = file_module(module_path)
    else:
        module = importlib.import_module(module_path)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise TypeError(f"fn {path!r} is not a function (got {fn!r})")
    return fn


def load_plugged_fn(
    name: str,
    config: DecoratedFunctionConfig,
    attr: str,
    decorated: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Build a task hook from a config entry: the imported async function — or, with
    no `fn`, the task's `decorated` method it overrides — wrapped under the plugged
    `name`, tagged like its `@vf.<attr>` equivalent. Unset metadata fields keep the
    wrapped function's tags."""
    if config.fn:
        fn = plugged_fn(config.fn)
    elif decorated is not None:
        fn = decorated
    else:
        raise ValueError(
            f"{attr} {name!r} plugs no `fn` and the task has no @vf.{attr} "
            f"method named {name!r} to override"
        )

    @functools.wraps(fn)
    def plugged(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    plugged.__name__ = name
    setattr(plugged, attr, True)
    priority = config.priority
    if priority is None:
        priority = getattr(fn, f"{attr}_priority", 0)
    setattr(plugged, f"{attr}_priority", priority)
    if isinstance(config, RewardFunctionConfig):
        weight = config.weight
        if weight is None:
            weight = getattr(fn, "_vf_weight", 1.0)
        plugged._vf_weight = weight
    return plugged


def taskset_config_type(taskset_id: str) -> type[TasksetConfig]:
    """Resolve the taskset's config specialization through its MRO."""
    return (
        concrete_type(taskset_class(taskset_id), TasksetConfig, origin=Taskset)
        or TasksetConfig
    )


def harness_config_type(harness_id: str) -> type[HarnessConfig]:
    """Resolve the harness's config specialization through its MRO."""
    return concrete_type(harness_class(harness_id), HarnessConfig) or HarnessConfig


def judge_config_type(judge_id: str) -> type[JudgeConfig]:
    """Resolve the judge's config specialization through its MRO."""
    return judge_config_cls(judge_class(judge_id))


def env_config_type(taskset_id: str, env_id: str = "") -> type[EnvConfig]:
    """Resolve the env's config specialization (`Env[YourConfig]`) through
    its MRO — `SingleAgentEnvConfig` for a plain taskset. The run's `env` field
    narrows to this, which is what gives `--env.<role>.model` addressing."""
    return (
        concrete_type(environment_class(taskset_id, env_id), EnvConfig, origin=Env)
        or EnvConfig
    )


def resolve_env_config(data: dict | EnvConfig | None) -> EnvConfig:
    """Narrow raw env-config data to the concrete env class's config type and
    validate. The one entry every consumer takes (CLI, TOML, the env-server wire),
    so role fields always validate against the real config class."""
    if isinstance(data, EnvConfig):
        cls = env_config_type(data.taskset.id, data.id)
        if isinstance(data, cls):
            return data  # already at least as specifically typed — keep
        data = data.model_dump()
    raw = dict(data or {})
    taskset = raw.get("taskset")
    taskset_id = (
        taskset.get("id", "")
        if isinstance(taskset, dict)
        else getattr(taskset, "id", "")
        if taskset is not None
        else ""
    )
    cls = env_config_type(taskset_id or "", raw.get("id") or "")
    return cls.model_validate(raw)


def task_type(taskset_id: str) -> type[Task]:
    """The taskset's `Task` subclass from its generic parameters — no data is
    loaded, so replay can cheaply recover the task type. Falls back to `Task`."""
    return taskset_class(taskset_id).task_type()
