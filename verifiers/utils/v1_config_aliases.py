from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def normalize_v1_config_aliases(
    env_config: Mapping[str, Any],
    *,
    args_key: str,
    global_harness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(env_config)
    taskset = normalized.pop("taskset", None)
    harness = normalized.pop("harness", None)
    if global_harness is None and taskset is None and harness is None:
        return normalized

    args = normalized.get(args_key, {}) or {}
    if not isinstance(args, Mapping):
        raise ValueError(f"{args_key} must be a table when using taskset/harness.")
    args = dict(args)

    config = args.get("config", {}) or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"{args_key}.config must be a table.")
    config = dict(config)

    if global_harness is not None:
        config["harness"] = merge_v1_config_tables(
            global_harness, config.get("harness"), "harness"
        )
    if taskset is not None:
        config["taskset"] = merge_v1_config_tables(
            config.get("taskset"), taskset, "taskset"
        )
    if harness is not None:
        config["harness"] = merge_v1_config_tables(
            config.get("harness"), harness, "harness"
        )

    args["config"] = config
    normalized[args_key] = args
    return normalized


def merge_v1_config_tables(
    base: object, overlay: object, field_name: str
) -> dict[str, Any]:
    base_table = v1_config_table(base, field_name)
    overlay_table = v1_config_table(overlay, field_name)
    merged = copy.deepcopy(base_table)
    for key, value in overlay_table.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_v1_config_tables(existing, value, field_name)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def v1_config_table(value: object, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table.")
    return value
