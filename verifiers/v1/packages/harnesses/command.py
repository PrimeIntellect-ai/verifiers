from collections.abc import Mapping
from typing import cast

from typing_extensions import NotRequired, TypedDict

from verifiers.clients import Client
from verifiers.types import ClientConfig, SamplingArgs

from ...config import SandboxConfig, sandbox_config_mapping
from ...types import (
    ConfigData,
    ConfigMap,
    Handler,
    ProgramCommand,
    ProgramMap,
    ProgramOptionMap,
    ProgramSetup,
    ProgramChannels,
)
from ...utils.binding_utils import Bindings
from ...utils.program_utils import program_list_items
from ...toolset import ToolsetCollection


DEFAULT_COMMAND_SANDBOX: ConfigData = {
    "image": "python:3.11-slim",
    "workdir": "/app",
    "scope": "rollout",
    "timeout_minutes": 120,
    "command_timeout": 900,
    "network_access": True,
}


class HarnessKwargs(TypedDict):
    user: NotRequired[Handler | str | ConfigMap | None]
    client: NotRequired[Client | ClientConfig | None]
    model: NotRequired[str | None]
    sampling_args: NotRequired[SamplingArgs | None]
    toolsets: NotRequired[ToolsetCollection | None]
    stops: NotRequired[list[Handler] | None]
    setups: NotRequired[list[Handler] | None]
    updates: NotRequired[list[Handler] | None]
    metrics: NotRequired[list[Handler] | None]
    rewards: NotRequired[list[Handler] | None]
    advantages: NotRequired[list[Handler] | None]
    cleanups: NotRequired[list[Handler] | None]


def command_program(
    *,
    command: ProgramCommand,
    sandbox: bool | ConfigMap | SandboxConfig,
    files: ProgramOptionMap | None = None,
    dirs: ProgramOptionMap | None = None,
    setup: ProgramSetup | None = None,
    bindings: Bindings | None = None,
    env: ProgramOptionMap | None = None,
    artifacts: ProgramOptionMap | None = None,
    channels: ProgramChannels | None = None,
    program: ProgramMap | None = None,
) -> ConfigData:
    config: ConfigData = {
        "command": command,
        "sandbox": sandbox is not False,
    }
    if files is not None:
        config["files"] = dict(files)
    if dirs is not None:
        config["dirs"] = dict(dirs)
    if setup is not None:
        config["setup"] = setup
    if bindings is not None:
        config["bindings"] = dict(bindings)
    if env is not None:
        config["env"] = dict(env)
    if artifacts is not None:
        config["artifacts"] = dict(artifacts)
    if channels is not None:
        config["channels"] = channels
    if program is not None:
        config = merge_program_defaults(config, program)
    return config


def command_sandbox(
    sandbox: bool | ConfigMap | SandboxConfig,
    defaults: ConfigMap | None = None,
) -> ConfigData | None:
    if sandbox is False:
        return None
    base = {**DEFAULT_COMMAND_SANDBOX, **dict(defaults or {})}
    if sandbox is True:
        return base
    return {**base, **(sandbox_config_mapping(sandbox) or {})}


def merge_program_defaults(defaults: ConfigMap, overrides: ProgramMap) -> ConfigData:
    merged = dict(defaults)
    for key, value in overrides.items():
        if (
            key in {"files", "dirs", "bindings", "env", "artifacts"}
            and isinstance(merged.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
            base = cast(ConfigMap, merged[key])
            patch = cast(ConfigMap, value)
            merged[key] = {**dict(base), **dict(patch)}
        elif key in {"setup", "args"}:
            merged[key] = [
                *program_list_items(merged.get(key), f"program.{key}"),
                *program_list_items(value, f"program.{key}"),
            ]
        else:
            merged[key] = value
    return merged
