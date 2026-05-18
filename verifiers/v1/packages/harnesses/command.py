from collections.abc import Mapping
from typing import TypeVar, cast

from pydantic import BaseModel

from ...config import HarnessConfig, SandboxConfig, sandbox_config_mapping
from ...harness import Harness
from ...types import (
    ConfigData,
    ConfigMap,
    ProgramCommand,
    ProgramChannels,
    ProgramMap,
    ProgramOptionMap,
    ProgramSetup,
    ProgramValue,
)
from ...utils.binding_utils import Bindings
from ...utils.config_utils import resolve_config_object
from ...utils.prompt_utils import (
    state_system_prompt_text,
    task_text as task_instruction_text,
)
from ...utils.program_utils import program_list_items

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


DEFAULT_COMMAND_SANDBOX: ConfigData = {
    "image": "python:3.11-slim",
    "workdir": "/app",
    "scope": "rollout",
    "timeout_minutes": 120,
    "command_timeout": 900,
    "network_access": True,
}


class CommandHarness(Harness[ConfigT]):
    def __init__(self, config: ConfigT | None = None):
        config = cast(ConfigT, self._coerce_config(config))
        super().__init__(config=config.model_copy(update={"program": None}))
        self.config = config
        sandbox = self.sandbox_value(config)
        self._configure_runtime(
            program=command_program(
                command=self.command(config),
                sandbox=sandbox,
                files=self.files(config),
                setup=self.setup(config),
                bindings=self.bindings_value(config),
                env=self.env(config),
                artifacts=self.artifacts(config),
                channels=self.channels(config),
                program=config.program,
            ),
            sandbox=command_sandbox(sandbox),
            system_prompt=config.system_prompt,
        )

    def command(self, config: ConfigT) -> ProgramCommand:
        raise NotImplementedError

    def sandbox_value(self, config: ConfigT) -> bool | ConfigMap | SandboxConfig:
        return config.sandbox if config.sandbox is not None else True

    def files(self, config: ConfigT) -> ProgramOptionMap:
        files: ProgramOptionMap = {}
        instruction_path = getattr(config, "instruction_path", None)
        system_prompt_path = getattr(config, "system_prompt_path", None)
        if instruction_path:
            files[str(instruction_path)] = cast(ProgramValue, task_instruction_text)
        if system_prompt_path and config.system_prompt is not None:
            files[str(system_prompt_path)] = cast(
                ProgramValue, state_system_prompt_text
            )
        return files

    def setup(self, config: ConfigT) -> ProgramSetup | None:
        return None

    def bindings_value(self, config: ConfigT) -> Bindings | None:
        return None

    def env(self, config: ConfigT) -> ProgramOptionMap | None:
        return None

    def artifacts(self, config: ConfigT) -> ProgramOptionMap | None:
        return None

    def channels(self, config: ConfigT) -> ProgramChannels | None:
        return None


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
    program: ProgramMap | BaseModel | str | None = None,
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
    resolved_program = resolve_config_object(program)
    if isinstance(resolved_program, BaseModel):
        resolved_program = resolved_program.model_dump(
            exclude_none=True,
            exclude_unset=True,
            exclude_defaults=True,
        )
    if resolved_program is not None:
        if not isinstance(resolved_program, Mapping):
            raise TypeError("program override must resolve to a mapping.")
        config = merge_program_defaults(config, cast(ProgramMap, resolved_program))
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
