from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from verifiers.clients import Client
from verifiers.types import ClientConfig, SamplingArgs

from ...harness import Harness


DEFAULT_CLI_SANDBOX = {
    "image": "python:3.11-slim",
    "workdir": "/app",
    "scope": "rollout",
    "timeout_minutes": 120,
    "command_timeout": 900,
    "network_access": True,
}


class CLIHarness(Harness):
    def __init__(
        self,
        command: str | list[object],
        *,
        sandbox: bool | Mapping[str, object] = True,
        files: Mapping[str, object] | None = None,
        dirs: Mapping[str, object] | None = None,
        setup: str | list[object] | None = None,
        env: Mapping[str, object] | None = None,
        artifacts: Mapping[str, object] | None = None,
        tools: str | None = None,
        program: Mapping[str, object] | None = None,
        system_prompt: object | None = None,
        user: object | None = None,
        client: Client | ClientConfig | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        max_turns: int | None = None,
        toolsets: object | None = None,
        stops: list[Callable[..., object]] | None = None,
        updates: list[Callable[..., object]] | None = None,
        metrics: list[Callable[..., object]] | None = None,
        rewards: list[Callable[..., object]] | None = None,
        advantages: list[Callable[..., object]] | None = None,
        cleanups: list[Callable[..., object]] | None = None,
        config: Mapping[str, object] | None = None,
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
        if env is not None:
            program_config["env"] = dict(env)
        if artifacts is not None:
            program_config["artifacts"] = dict(artifacts)
        if tools is not None:
            program_config["tools"] = tools
        if program is not None:
            program_config = merge_program_defaults(program_config, program)
        sandbox_config = DEFAULT_CLI_SANDBOX if sandbox is True else None
        if isinstance(sandbox, Mapping):
            sandbox_config = {**DEFAULT_CLI_SANDBOX, **dict(sandbox)}
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
            key in {"files", "dirs", "env", "artifacts"}
            and isinstance(merged.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
            base = cast(Mapping[str, object], merged[key])
            patch = cast(Mapping[str, object], value)
            merged[key] = {**dict(base), **dict(patch)}
        elif key in {"setup", "args"}:
            merged[key] = [*list_items(merged.get(key)), *list_items(value)]
        else:
            merged[key] = value
    return merged


def list_items(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return list(value)
    raise TypeError("program setup/args values must be strings or lists.")
