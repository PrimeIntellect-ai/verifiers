from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def merge_v1_config_aliases(
    *,
    taskset: object | None = None,
    harness: object | None = None,
    global_harness: object | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if global_harness is None and taskset is None and harness is None:
        return config

    if global_harness is not None:
        config["harness"] = merge_v1_config_tables(global_harness, harness, "harness")
    elif harness is not None:
        config["harness"] = merge_v1_config_tables(None, harness, "harness")

    if taskset is not None:
        config["taskset"] = merge_v1_config_tables(None, taskset, "taskset")

    return config


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


def v1_config_table(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a table.")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings.")
        result[key] = item
    return result
