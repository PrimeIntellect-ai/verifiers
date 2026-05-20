import functools
import inspect
from collections.abc import Iterable, Mapping
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel

from .config_utils import resolve_config_object
from ..types import ConfigMap, Handler

CallableKind: TypeAlias = Literal[
    "stop", "setup", "update", "metric", "reward", "advantage", "cleanup", "teardown"
]

CALLABLE_KIND_FIELDS: dict[CallableKind, str] = {
    "stop": "stops",
    "setup": "setups",
    "update": "updates",
    "metric": "metrics",
    "reward": "rewards",
    "advantage": "advantages",
    "cleanup": "cleanups",
    "teardown": "teardowns",
}


def merge_config_callables(
    values: Iterable[Handler],
    config: object,
    kind: CallableKind,
) -> list[Handler]:
    return [*config_callables(values, kind), *config_callables(config, kind)]


def merge_config_handler_map(
    values: dict[CallableKind, Iterable[Handler]],
    config: object,
) -> dict[CallableKind, list[Handler]]:
    return {
        kind: merge_config_callables(
            constructor_values, getattr(config, CALLABLE_KIND_FIELDS[kind]), kind
        )
        for kind, constructor_values in values.items()
    }


def config_callables(value: object, kind: CallableKind) -> list[Handler]:
    if value is None:
        return []
    if isinstance(value, str):
        return [callable_config_item(value, kind)]
    if isinstance(value, Mapping):
        return [callable_config_item(value, kind)]
    if isinstance(value, Iterable):
        return [callable_config_item(item, kind) for item in value]
    return [callable_config_item(value, kind)]


def callable_config_item(value: object, kind: CallableKind) -> Handler:
    value = resolve_config_object(value)
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return callable_from_mapping(cast(ConfigMap, value), kind)
    if not callable(value):
        raise TypeError(f"{kind} config entries must resolve to callables.")
    return cast(Handler, value)


def callable_from_mapping(spec: ConfigMap, kind: CallableKind) -> Handler:
    allowed = callable_config_keys(kind)
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"{kind} callable config has unknown keys: {sorted(unknown)}.")
    if bool(spec.get("skip", False)):
        raise ValueError(
            f"{kind} callable config should be removed instead of skipped."
        )
    fn = resolve_config_object(spec.get("fn"))
    if not callable(fn):
        raise TypeError(f"{kind} callable config requires callable fn.")
    metadata = {key: spec[key] for key in spec if key not in {"fn", "skip"}}
    return configured_callable(cast(Handler, fn), kind, metadata)


def callable_config_keys(kind: CallableKind) -> set[str]:
    keys = {"fn", "priority", "skip"}
    if kind in {"update", "metric", "reward", "cleanup"}:
        keys.add("stage")
    if kind == "reward":
        keys.add("weight")
    return keys


def configured_callable(
    fn: Handler,
    kind: CallableKind,
    metadata: ConfigMap,
) -> Handler:
    if not metadata:
        return fn

    @functools.wraps(fn)
    async def wrapper(**kwargs: object) -> object:
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    setattr(wrapper, "__signature__", inspect.signature(fn))
    setattr(wrapper, kind, True)
    if "priority" in metadata:
        priority = metadata["priority"]
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError(f"{kind} priority must be an integer.")
        setattr(wrapper, f"{kind}_priority", priority)
    if "stage" in metadata:
        stage = metadata["stage"]
        if stage not in {"rollout", "group"}:
            raise ValueError(f"{kind} stage must be 'rollout' or 'group'.")
        setattr(wrapper, f"{kind}_stage", stage)
    if "weight" in metadata:
        weight = metadata["weight"]
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise TypeError("reward weight must be numeric.")
        setattr(wrapper, "reward_weight", float(weight))
    return cast(Handler, wrapper)
