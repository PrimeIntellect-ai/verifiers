from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from verifiers.rubrics.rubric import Rubric
from verifiers.rubrics.rubric_group import RubricGroup
from verifiers.types import GroupRewardFunc, RewardFunc
from verifiers.utils.async_utils import maybe_await

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    ResourcePatch,
)

RESERVED_FUNC_KEYS = {"fn", "name", "enabled", "weight"}
RESERVED_RUBRIC_KEYS = {"rubric", "name", "enabled"}


def canonicalize_rubric_config(
    config: ChannelConfig,
) -> dict[str, list[dict[str, Any]]]:
    if config is None:
        return empty_rubric_config()
    if isinstance(config, str) or is_rubric_object(config):
        canonical = empty_rubric_config()
        canonical["rubrics"] = [rubric_entry(config)]
        return canonical
    if callable(config):
        canonical = empty_rubric_config()
        canonical["rewards"] = [function_entry(config, 1.0)]
        return canonical
    if isinstance(config, Sequence) and not isinstance(config, str | bytes):
        return canonicalize_top_level_sequence(config)
    if isinstance(config, Mapping):
        config = cast(Mapping[str, Any], config)
        if "fn" in config:
            canonical = empty_rubric_config()
            canonical["rewards"] = [function_entry_from_mapping(config, 1.0)]
            return canonical
        if "rubric" in config:
            canonical = empty_rubric_config()
            canonical["rubrics"] = [rubric_entry_from_mapping(config)]
            return canonical
        canonical = {
            "rewards": canonicalize_entry_list(config.get("rewards", []), "fn", 1.0),
            "metrics": canonicalize_entry_list(config.get("metrics", []), "fn", 0.0),
            "rubrics": canonicalize_entry_list(
                config.get("rubrics", []), "rubric", None
            ),
            "cleanup": cleanup_entries(config.get("cleanup", [])),
        }
        return canonical
    raise TypeError(f"Unsupported rubric channel config: {config!r}")


def empty_rubric_config() -> dict[str, list[dict[str, Any]]]:
    return {"rewards": [], "metrics": [], "rubrics": [], "cleanup": []}


def canonicalize_top_level_sequence(
    config: Sequence[object],
) -> dict[str, list[dict[str, Any]]]:
    if not config:
        return empty_rubric_config()
    if all(isinstance(item, str) for item in config):
        canonical = empty_rubric_config()
        canonical["rewards"] = [function_entry(item, 1.0) for item in config]
        return canonical
    if all(is_rubric_object(item) for item in config):
        canonical = empty_rubric_config()
        canonical["rubrics"] = [rubric_entry(item) for item in config]
        return canonical
    if all(isinstance(item, Mapping) and "fn" in item for item in config):
        canonical = empty_rubric_config()
        canonical["rewards"] = [
            function_entry_from_mapping(cast(Mapping[str, Any], item), 1.0)
            for item in config
        ]
        return canonical
    if all(isinstance(item, Mapping) and "rubric" in item for item in config):
        canonical = empty_rubric_config()
        canonical["rubrics"] = [
            rubric_entry_from_mapping(cast(Mapping[str, Any], item)) for item in config
        ]
        return canonical
    return merge_canonical_rubric_configs(
        [canonicalize_rubric_config(item) for item in config]
    )


def merge_canonical_rubric_configs(
    configs: Sequence[dict[str, list[dict[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    canonical = empty_rubric_config()
    for config in configs:
        for key in canonical:
            canonical[key].extend(config[key])
    return canonical


def canonicalize_entry_list(
    raw_entries: object,
    kind: str,
    default_weight: float | None,
) -> list[dict[str, Any]]:
    if raw_entries is None:
        return []
    entries = raw_entries if isinstance(raw_entries, list | tuple) else [raw_entries]
    if not entries:
        return []
    if all(isinstance(entry, str) or callable(entry) for entry in entries):
        if kind == "fn":
            return [function_entry(entry, default_weight) for entry in entries]
        return [rubric_entry(entry) for entry in entries]
    if all(isinstance(entry, Mapping) for entry in entries):
        if kind == "fn":
            return [
                function_entry_from_mapping(
                    cast(Mapping[str, Any], entry), default_weight
                )
                for entry in entries
            ]
        return [
            rubric_entry_from_mapping(cast(Mapping[str, Any], entry))
            for entry in entries
        ]
    raise TypeError(
        "Rubric entry lists must contain either only names/objects or only dicts."
    )


def function_entry(ref: object, default_weight: float | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"fn": ref}
    if default_weight is not None:
        entry["weight"] = default_weight
    return entry


def function_entry_from_mapping(
    raw: Mapping[str, Any], default_weight: float | None
) -> dict[str, Any]:
    if "fn" not in raw:
        raise ValueError("Rubric reward/metric entries must include 'fn'.")
    entry = dict(raw)
    if default_weight is not None:
        entry.setdefault("weight", default_weight)
    return entry


def rubric_entry(ref: object) -> dict[str, Any]:
    return {"rubric": ref}


def rubric_entry_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "rubric" not in raw:
        raise ValueError("Rubric object entries must include 'rubric'.")
    return dict(raw)


def cleanup_entries(raw_entries: object) -> list[dict[str, Any]]:
    if raw_entries is None:
        return []
    entries = raw_entries if isinstance(raw_entries, list | tuple) else [raw_entries]
    if all(isinstance(entry, Mapping) and "cleanup" in entry for entry in entries):
        return [dict(cast(Mapping[str, Any], entry)) for entry in entries]
    return [{"cleanup": entry} for entry in entries]


def resolve_rubric_channel(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    return ResourcePatch(objects={"rubric": build_rubric(configs, context)})


def build_rubric(configs: list[ChannelConfig], context: ChannelContext) -> Rubric:
    canonical_configs = [canonicalize_rubric_config(config) for config in configs]
    reward_entries = [
        entry for config in canonical_configs for entry in config["rewards"]
    ]
    metric_entries = [
        entry for config in canonical_configs for entry in config["metrics"]
    ]
    rubric_entries = [
        entry for config in canonical_configs for entry in config["rubrics"]
    ]
    cleanup_entries_ = [
        entry for config in canonical_configs for entry in config["cleanup"]
    ]
    funcs: list[RewardFunc | GroupRewardFunc] = []
    weights: list[float] = []
    seen_names: set[str] = set()

    for entry in [*reward_entries, *metric_entries]:
        if entry.get("enabled", True) is False:
            continue
        fn = resolve_function(entry["fn"], context)
        name = entry_name(entry, fn)
        if name in seen_names:
            raise ValueError(
                f"Rubric entry {name!r} is configured more than once. Set a "
                "unique 'name' when reusing the same function."
            )
        seen_names.add(name)
        funcs.append(bind_entry_metadata(fn, entry, name))
        weights.append(float(entry.get("weight", 1.0)))

    rubrics: list[Rubric] = []
    if funcs:
        rubrics.append(Rubric(funcs=funcs, weights=weights))
    for entry in rubric_entries:
        if entry.get("enabled", True) is False:
            continue
        rubrics.append(resolve_rubric(entry, context))

    if not rubrics:
        rubric = NoOpRubric()
    elif len(rubrics) == 1:
        rubric = rubrics[0]
    else:
        rubric = RubricGroup(rubrics)
    for entry in cleanup_entries_:
        cleanup = entry["cleanup"]
        if not callable(cleanup):
            raise TypeError("Rubric cleanup entries must be callable.")
        rubric.add_cleanup_handler(cleanup)
    return rubric


def extend_rubric_channel(
    current: object, configs: list[ChannelConfig], context: ChannelContext
) -> Rubric:
    if not isinstance(current, Rubric):
        raise TypeError("Resolved rubric must be a Rubric.")
    incoming = build_rubric(configs, context)
    if isinstance(current, NoOpRubric):
        return incoming
    if isinstance(incoming, NoOpRubric):
        return current
    return compose_rubrics(current, incoming)


def resolve_function(
    ref: object, context: ChannelContext
) -> RewardFunc | GroupRewardFunc:
    obj = context.get_object(ref)
    if not callable(obj) or is_rubric_object(obj):
        raise TypeError(f"Rubric function {ref!r} did not resolve to a function.")
    return cast(RewardFunc | GroupRewardFunc, obj)


def resolve_rubric(entry: Mapping[str, Any], context: ChannelContext) -> Rubric:
    obj = context.get_object(entry["rubric"])
    kwargs = {
        key: value for key, value in entry.items() if key not in RESERVED_RUBRIC_KEYS
    }
    if isinstance(obj, Rubric):
        if kwargs:
            raise ValueError("Rubric instances cannot receive constructor metadata.")
        return obj
    if inspect.isclass(obj) and issubclass(obj, Rubric):
        return obj(**kwargs)
    raise TypeError(f"Rubric object {entry['rubric']!r} did not resolve to a Rubric.")


def bind_entry_metadata(
    fn: RewardFunc | GroupRewardFunc, entry: Mapping[str, Any], name: str
) -> RewardFunc | GroupRewardFunc:
    metadata = {
        key: value for key, value in entry.items() if key not in RESERVED_FUNC_KEYS
    }
    if not metadata and name == getattr(fn, "__name__", name):
        return fn
    signature = inspect.signature(fn)

    async def wrapped(**objects):
        kwargs = dict(metadata)
        kwargs.update(objects)
        if not any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
        return await maybe_await(fn, **kwargs)

    wrapped.__name__ = name
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__signature__ = signature  # type: ignore[attr-defined]
    return cast(RewardFunc | GroupRewardFunc, wrapped)


def entry_name(entry: Mapping[str, Any], fn: Callable[..., object]) -> str:
    raw_name = entry.get("name") or getattr(fn, "__name__", None)
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError("Rubric entries require a stable name.")
    return raw_name


def is_rubric_object(obj: object) -> bool:
    return isinstance(obj, Rubric) or (inspect.isclass(obj) and issubclass(obj, Rubric))


class NoOpRubric(Rubric):
    async def score_rollout(self, state):
        await self.dummy_score_rollout(state)

    async def score_group(self, states):
        await self.dummy_score_group(states)


def compose_rubrics(*rubrics: Rubric | None) -> Rubric:
    concrete = [rubric for rubric in rubrics if rubric is not None]
    if not concrete:
        return NoOpRubric()
    if len(concrete) == 1:
        return concrete[0]
    return RubricGroup(concrete)


def attach_resources(rubric: Rubric, resources: object) -> None:
    if isinstance(rubric, RubricGroup):
        for child in rubric.rubrics:
            attach_resources(child, resources)
        return
    rubric.add_class_object("resources", resources)


rubric_channel = Channel(
    name="rubric",
    outputs={"rubric": Rubric},
    always_resolve=True,
    canonicalize_fn=canonicalize_rubric_config,
    resolve_fn=resolve_rubric_channel,
    extend_fn=extend_rubric_channel,
)
