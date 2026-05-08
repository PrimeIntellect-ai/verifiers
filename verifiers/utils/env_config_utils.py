from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast


def config_table(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a table.")
    result: dict[str, object] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError(f"{field} keys must be strings.")
        result[key] = item
    return result


def merge_config_tables(base: object, overlay: object, field: str) -> dict[str, object]:
    merged = copy.deepcopy(config_table(base, field))
    for key, value in config_table(overlay, field).items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_config_tables(existing, value, field)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_env_config_sections(
    raw: Mapping[str, object], *, global_harness: object | None = None
) -> dict[str, object]:
    config = dict(raw)
    env_args = config_table(config.pop("env_args", {}), "env_args")
    args = config_table(config.pop("args", {}), "args")
    overlap = set(env_args) & set(args)
    if overlap:
        raise ValueError(
            f"Environment arg key(s) {overlap} appear in both args and env_args."
        )
    env_args = {**env_args, **args}

    legacy_config = config_table(env_args.pop("config", {}), "env_args.config")
    taskset = merge_config_tables(
        legacy_config.get("taskset"), config.pop("taskset", None), "taskset"
    )
    harness = merge_config_tables(
        merge_config_tables(global_harness, legacy_config.get("harness"), "harness"),
        config.pop("harness", None),
        "harness",
    )

    if taskset:
        config["taskset"] = taskset
    if harness:
        config["harness"] = harness
    if env_args:
        config["env_args"] = env_args
    return config
