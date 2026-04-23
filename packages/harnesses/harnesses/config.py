from __future__ import annotations

import importlib
import inspect
from typing import Any


def build_harness_from_config(
    config: Any | None,
    *,
    agent_workdir: str | None = None,
):
    """Build or pass through a Harness from TOML-friendly config.

    Supported config shapes:

    - `Harness(...)`
    - `"openclaw"`
    - `{agent = "openclaw", cwd = "/app"}`
    - `{name = "codex", reasoning_effort = "medium"}`
    - `{factory = "my_pkg.my_module.make_harness", ...}`
    - `{harness = {factory = "my_pkg.my_module.make_harness", ...}}`
    - `{agent = {harness = {...}, timeout_sec = 3600}}`

    The lookup is convention-based: `agent = "openclaw"` imports
    `harnesses.openclaw` and calls `openclaw_harness`.
    """
    harness_config: dict[str, Any] = {}
    if isinstance(config, str):
        harness_config = {"harness": config}
    elif isinstance(config, dict):
        harness_config = dict(config)
        agent_config = config.get("agent")
        if isinstance(agent_config, dict) and isinstance(
            agent_config.get("harness"), dict
        ):
            harness_config = dict(agent_config["harness"])
            harness_config.setdefault(
                "timeout_seconds", agent_config.get("timeout_sec")
            )
    elif config is not None:
        return config

    harness_value = harness_config.pop("harness", None)
    if isinstance(harness_value, dict):
        harness_config = {**harness_config, **harness_value}
        harness_value = harness_config.pop("harness", None)

    factory_path = harness_config.pop("factory", None)
    harness_name = harness_value
    config_name = harness_config.pop("name", None)
    agent_name = harness_config.pop("agent", None)
    name = str(
        factory_path or harness_name or config_name or agent_name or "openclaw"
    ).strip()

    harness_config.pop("transport", None)
    if factory_path:
        module_path, attr = str(factory_path).rsplit(".", 1)
        factory = getattr(importlib.import_module(module_path), attr)
    else:
        module_name = name.replace("-", "_")
        module = importlib.import_module(f"harnesses.{module_name}")
        factory = getattr(module, f"{module_name}_harness")

    normalized_name = name.replace("-", "_")
    prefixes = {f"{normalized_name}_", "agent_"}
    first_name_part = normalized_name.split("_", 1)[0]
    if first_name_part != normalized_name:
        prefixes.add(f"{first_name_part}_")

    for key, value in list(harness_config.items()):
        for prefix in prefixes:
            if key.startswith(prefix) and key != "agent_workdir":
                harness_config.setdefault(key.removeprefix(prefix), value)
                break

    if "cwd" in harness_config and "agent_workdir" not in harness_config:
        harness_config["agent_workdir"] = harness_config.pop("cwd")
    if "workdir" in harness_config and "agent_workdir" not in harness_config:
        harness_config["agent_workdir"] = harness_config.pop("workdir")
    if "timeout_sec" in harness_config and "timeout_seconds" not in harness_config:
        harness_config["timeout_seconds"] = harness_config.pop("timeout_sec")
    if agent_workdir and "agent_workdir" not in harness_config:
        harness_config["agent_workdir"] = agent_workdir

    signature = inspect.signature(factory)
    accepts_kwargs = any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return factory(**harness_config)

    kwargs = {
        key: value
        for key, value in harness_config.items()
        if key in signature.parameters and value is not None
    }
    return factory(**kwargs)
