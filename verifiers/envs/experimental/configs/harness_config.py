from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from verifiers.envs.experimental.channels.sandbox_channel import (
    SandboxSpec,
    SandboxTimeouts,
)
from verifiers.types import ClientType


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class RunConfig(Config):
    max_turns: int = 50
    parallel_model_requests: bool = False
    error_formatter: Callable[[Exception], str] = str
    stop_errors: tuple[type[Exception], ...] = ()
    max_tool_calls_per_turn: int | None = None
    tool_call_limit_message: str = (
        "Please provide at most {limit} tool call(s). Found {count}."
    )


class EndpointConfig(Config):
    port: int | None = None
    url: str | None = None
    secret: str | None = None
    api_client_type: ClientType = "openai_chat_completions"
    poll_interval: float = 1.0
    use_tunnel: bool = False


class CliPaths(Config):
    instruction: str = "/task/instruction.md"
    system_prompt: str = "/task/system_prompt.md"
    log: str | None = None


class CliMetrics(Config):
    path: str | None = None
    prefix: str = ""
    key: str | None = None
    keys: tuple[str, ...] | None = None
    tool_names: tuple[str, ...] = ()


class CliConfig(Config):
    command: str
    workdir: str = "/workspace"
    paths: CliPaths = Field(default_factory=CliPaths)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 3600.0
    metrics: CliMetrics = Field(default_factory=CliMetrics)


class SandboxSetup(Config):
    uploads: dict[str, object] = Field(default_factory=dict)
    upload_mapping: dict[str, str] = Field(default_factory=dict)
    skills_path: str | None = None
    commands: tuple[str, ...] = ()
    install_command: str | None = None
    install_timeout: int = 300
    install_env: dict[str, str] | None = None
    post_install_uploads: dict[str, str] = Field(default_factory=dict)
    post_install_command: str | None = None


class SandboxRuntime(Config):
    wait_for_creation_max_attempts: int = 120
    creations_per_minute: float | None = 128
    client_max_workers: int = 50
    client_max_connections: int = 1000
    client_max_keepalive_connections: int = 200
    max_retries: int = 5
    base_delay: float = 0.5
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 30.0
    jitter: float = 1e-3
    timeouts: SandboxTimeouts = SandboxTimeouts()


class SandboxScoring(Config):
    retain: bool = False


class SandboxConfig(Config):
    spec: SandboxSpec = Field(default_factory=SandboxSpec)
    setup: SandboxSetup = Field(default_factory=SandboxSetup)
    runtime: SandboxRuntime = Field(default_factory=SandboxRuntime)
    scoring: SandboxScoring = Field(default_factory=SandboxScoring)
