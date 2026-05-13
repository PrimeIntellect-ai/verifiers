from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from verifiers.clients import Client
from verifiers.types import ClientConfig, SamplingArgs
from typing_extensions import NotRequired, TypedDict

from ...config import HarnessConfig, SandboxConfig, sandbox_config_mapping
from ...harness import Harness
from ...types import ConfigMap, Handler, PromptInput
from ...utils.program_utils import program_list_items


DEFAULT_CLI_SANDBOX = {
    "image": "python:3.11-slim",
    "workdir": "/app",
    "scope": "rollout",
    "timeout_minutes": 120,
    "command_timeout": 900,
    "network_access": True,
}


class CLIHarnessKwargs(TypedDict):
    user: NotRequired[Handler | str | ConfigMap | None]
    client: NotRequired[Client | ClientConfig | None]
    model: NotRequired[str | None]
    sampling_args: NotRequired[SamplingArgs | None]
    toolsets: NotRequired[object | None]
    stops: NotRequired[list[Callable[..., object]] | None]
    setups: NotRequired[list[Callable[..., object]] | None]
    updates: NotRequired[list[Callable[..., object]] | None]
    metrics: NotRequired[list[Callable[..., object]] | None]
    rewards: NotRequired[list[Callable[..., object]] | None]
    advantages: NotRequired[list[Callable[..., object]] | None]
    cleanups: NotRequired[list[Callable[..., object]] | None]


class CLIHarness(Harness):
    def __init__(
        self,
        command: str | list[object],
        *,
        sandbox: bool | Mapping[str, object] | SandboxConfig = True,
        files: Mapping[str, object] | None = None,
        dirs: Mapping[str, object] | None = None,
        setup: object | list[object] | None = None,
        bindings: Mapping[str, object] | None = None,
        env: Mapping[str, object] | None = None,
        artifacts: Mapping[str, object] | None = None,
        tools: object | None = None,
        program: Mapping[str, object] | None = None,
        system_prompt: PromptInput | None = None,
        user: Handler | str | ConfigMap | None = None,
        client: Client | ClientConfig | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        max_turns: int | None = None,
        toolsets: object | None = None,
        stops: list[Callable[..., object]] | None = None,
        setups: list[Callable[..., object]] | None = None,
        updates: list[Callable[..., object]] | None = None,
        metrics: list[Callable[..., object]] | None = None,
        rewards: list[Callable[..., object]] | None = None,
        advantages: list[Callable[..., object]] | None = None,
        cleanups: list[Callable[..., object]] | None = None,
        config: HarnessConfig | Mapping[str, object] | None = None,
    ):
        program_config: dict[str, object] = {
            "command": command,
            "sandbox": sandbox,
        }
        if files is not None:
            program_config["files"] = dict(files)
        if dirs is not None:
            program_config["dirs"] = dict(dirs)
        if setup is not None:
            program_config["setup"] = setup
        if bindings is not None:
            program_config["bindings"] = dict(bindings)
        if env is not None:
            program_config["env"] = dict(env)
        if artifacts is not None:
            program_config["artifacts"] = dict(artifacts)
        if tools is not None:
            program_config["tools"] = tools
        if program is not None:
            program_config = merge_program_defaults(program_config, program)
        sandbox_config = DEFAULT_CLI_SANDBOX if sandbox is True else None
        if sandbox is not True and sandbox is not False:
            sandbox_config = {
                **DEFAULT_CLI_SANDBOX,
                **(sandbox_config_mapping(sandbox) or {}),
            }
        super().__init__(
            program=program_config,
            system_prompt=system_prompt,
            user=user,
            sandbox=sandbox_config,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_turns=max_turns,
            toolsets=toolsets,
            stops=stops,
            setups=setups,
            updates=updates,
            metrics=metrics,
            rewards=rewards,
            advantages=advantages,
            cleanups=cleanups,
            config=config,
        )


def merge_program_defaults(
    defaults: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if (
            key in {"files", "dirs", "bindings", "env", "artifacts"}
            and isinstance(merged.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
            base = cast(Mapping[str, object], merged[key])
            patch = cast(Mapping[str, object], value)
            merged[key] = {**dict(base), **dict(patch)}
        elif key in {"setup", "args"}:
            merged[key] = [
                *program_list_items(merged.get(key), f"program.{key}"),
                *program_list_items(value, f"program.{key}"),
            ]
        else:
            merged[key] = value
    return merged
