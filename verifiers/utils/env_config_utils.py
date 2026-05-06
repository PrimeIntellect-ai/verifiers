from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def config_table(value: object, field: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a table.")
    return dict(cast(Mapping[str, object], value))


def normalize_env_config_sections(raw: Mapping[str, object]) -> dict[str, object]:
    config = dict(raw)
    env_args = config_table(config.pop("env_args", {}), "env_args")
    args = config_table(config.pop("args", {}), "args")
    overlap = set(env_args) & set(args)
    if overlap:
        raise ValueError(
            f"Environment arg key(s) {overlap} appear in both args and env_args."
        )
    env_args = {**env_args, **args}

    taskset = config.pop("taskset", None)
    harness = config.pop("harness", None)
    child_config: dict[str, object] = {}
    if taskset is not None:
        child_config["taskset"] = config_table(taskset, "taskset")
    if harness is not None:
        child_config["harness"] = config_table(harness, "harness")

    if child_config:
        existing_config = config_table(env_args.get("config", {}), "env_args.config")
        overlap = set(existing_config) & set(child_config)
        if overlap:
            raise ValueError(
                f"Environment config section(s) {overlap} appear in both config and top-level sections."
            )
        env_args["config"] = {**existing_config, **child_config}

    if env_args:
        config["env_args"] = env_args
    return config
