from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import httpx
import tenacity as tc
from aiolimiter import AsyncLimiter
from prime_sandboxes import (
    AdvancedConfigs,
    APIError,
    CommandTimeoutError,
    DownloadTimeoutError,
    SandboxOOMError,
    SandboxTimeoutError,
    UploadTimeoutError,
)

from verifiers.envs.experimental.channels.channel import (
    Channel,
    ChannelConfig,
    ChannelContext,
    LifecycleHooks,
    ResourcePatch,
)


@dataclass(frozen=True)
class SandboxTimeouts:
    read_file: int = 10
    extract: int = 60
    poll: int = 60
    mkdir: int = 10
    install: int = 300


@dataclass(frozen=True)
class SandboxSpec:
    image: str = "python:3.11-slim"
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 5
    gpu_count: int = 0
    gpu_type: str | None = None
    vm: bool | None = None
    network_access: bool = True
    timeout_minutes: int = 60
    start_command: str = "tail -f /dev/null"
    environment_vars: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] | None = None
    team_id: str | None = None
    advanced_configs: AdvancedConfigs | None = None
    registry_credentials_id: str | None = None
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SandboxSeed:
    image: str | None = None
    cpu_cores: int | None = None
    memory_gb: int | None = None
    disk_size_gb: int | None = None
    gpu_count: int | None = None
    gpu_type: str | None = None
    vm: bool | None = None
    network_access: bool | None = None
    timeout_minutes: int | None = None
    start_command: str | None = None
    environment_vars: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] | None = None
    team_id: str | None = None
    advanced_configs: AdvancedConfigs | None = None
    registry_credentials_id: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    mounts: dict[str, str] = field(default_factory=dict)
    setup_commands: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def is_retryable_sandbox_api_error(exception: BaseException) -> bool:
    if not isinstance(exception, APIError):
        return False
    error = str(exception)
    return any(
        token in error
        for token in (
            "502",
            "503",
            "ConnectError",
            "Temporary failure in name resolution",
        )
    )


def is_retryable_sandbox_read_error(exception: BaseException) -> bool:
    return isinstance(
        exception,
        (
            httpx.ReadTimeout,
            CommandTimeoutError,
            UploadTimeoutError,
            DownloadTimeoutError,
        ),
    ) or is_retryable_sandbox_api_error(exception)


class SandboxResources:
    """Environment-scoped sandbox client, retry policy, and active sandbox tracking."""

    def __init__(
        self,
        *,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        client_max_workers: int = 50,
        client_max_connections: int = 1000,
        client_max_keepalive_connections: int = 200,
        creations_per_minute: float | None = 128,
        timeouts: SandboxTimeouts = SandboxTimeouts(),
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.timeouts = timeouts
        self.client_max_workers = client_max_workers
        self.client_max_connections = client_max_connections
        self.client_max_keepalive_connections = client_max_keepalive_connections
        self.creation_rate_limiter = (
            AsyncLimiter(max_rate=creations_per_minute, time_period=60.0)
            if creations_per_minute is not None
            else None
        )
        self.with_retry: Callable = tc.AsyncRetrying(
            retry=tc.retry_if_exception(is_retryable_sandbox_read_error),
            stop=tc.stop_after_attempt(max_retries + 1),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        ).wraps
        self.active_sandboxes: set[str] = set()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from verifiers.utils.threaded_sandbox_client import (
                ThreadedAsyncSandboxClient,
            )

            self._client = ThreadedAsyncSandboxClient(
                max_workers=self.client_max_workers,
                max_connections=self.client_max_connections,
                max_keepalive_connections=self.client_max_keepalive_connections,
            )
        return self._client

    async def create(self, request: Any, *, max_attempts: int) -> str:
        if self.creation_rate_limiter is not None:
            await self.creation_rate_limiter.acquire()
        sandbox = await self.with_retry(self.client.create)(request)
        self.active_sandboxes.add(sandbox.id)
        await self.client.wait_for_creation(sandbox.id, max_attempts=max_attempts)
        return sandbox.id

    async def delete(self, sandbox_id: str) -> None:
        await self.with_retry(self.client.delete)(sandbox_id)
        self.active_sandboxes.discard(sandbox_id)

    async def bulk_delete_active(self) -> None:
        if not self.active_sandboxes:
            return
        sandbox_ids = list(self.active_sandboxes)
        try:
            if hasattr(self.client, "bulk_delete"):
                await self.with_retry(self.client.bulk_delete)(sandbox_ids)
            else:
                for sandbox_id in sandbox_ids:
                    await self.with_retry(self.client.delete)(sandbox_id)
            self.active_sandboxes.clear()
        except Exception as e:
            self.logger.warning(f"Failed to bulk-delete active sandboxes: {e}")

    def retain_for_scoring(self, sandbox_id: str) -> None:
        self.active_sandboxes.discard(sandbox_id)

    async def teardown(self) -> None:
        await self.bulk_delete_active()
        if self._client is not None and hasattr(self._client, "teardown"):
            self._client.teardown()


SANDBOX_FAILURES = (SandboxOOMError, SandboxTimeoutError)


def resolve_sandbox(
    configs: list[ChannelConfig], context: ChannelContext
) -> ResourcePatch:
    merged: dict[str, object] = {}
    for config in configs:
        if config is None:
            continue
        if not isinstance(config, Mapping):
            config = {"spec": config}
        for key, value in config.items():
            key = str(key)
            if key == "spec" and key in merged:
                merged[key] = merge_sandbox_spec(merged[key], value)
                continue
            if key == "uploads" and key in merged:
                merged[key] = merge_sandbox_uploads(merged[key], value)
                continue
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"Sandbox channel key {key!r} received multiple values."
                )
            merged[key] = value
    objects: dict[str, object] = {"sandbox_scoring": False, "sandbox_uploads": {}}
    hooks = LifecycleHooks()
    if "spec" in merged:
        objects["sandbox_request"] = merged["spec"]
    if "scoring" in merged:
        objects["sandbox_scoring"] = bool(merged["scoring"])
    if "uploads" in merged:
        objects["sandbox_uploads"] = sandbox_uploads(merged["uploads"])
    if "runtime" in merged:
        if context.phase != "env":
            return ResourcePatch(objects=objects)
        runtime_patch = materialize_sandbox(merged["runtime"], context)
        objects.update(runtime_patch.objects)
        hooks = hooks.merged(runtime_patch.hooks)
    return ResourcePatch(objects=objects, hooks=hooks)


def materialize_sandbox(config: object, context: ChannelContext) -> ResourcePatch:
    if config is None:
        return ResourcePatch()
    if not isinstance(config, Mapping):
        raise TypeError("The sandbox runtime config must resolve to a mapping.")
    config = cast(Mapping[str, object], config)
    sandbox_resources = SandboxResources(
        max_retries=as_int(config.get("max_retries", 5)),
        base_delay=as_float(config.get("base_delay", 0.5)),
        backoff_factor=as_float(config.get("backoff_factor", 2.0)),
        max_backoff_seconds=as_float(config.get("max_backoff_seconds", 30.0)),
        jitter=as_float(config.get("jitter", 1e-3)),
        client_max_workers=as_int(config.get("client_max_workers", 50)),
        client_max_connections=as_int(config.get("client_max_connections", 1000)),
        client_max_keepalive_connections=as_int(
            config.get("client_max_keepalive_connections", 200)
        ),
        creations_per_minute=optional_float(config.get("creations_per_minute", 128)),
        timeouts=sandbox_timeouts(config.get("timeouts")),
        logger=channel_logger(context),
    )
    return ResourcePatch(
        objects={"sandbox_runtime": sandbox_resources},
        hooks=LifecycleHooks(teardown=(sandbox_resources.teardown,)),
    )


sandbox_channel = Channel(
    name="sandbox",
    outputs={
        "sandbox_request": object,
        "sandbox_runtime": SandboxResources,
        "sandbox_scoring": bool,
        "sandbox_uploads": dict,
    },
    resolve_fn=resolve_sandbox,
)


def merge_sandbox_spec(existing: object, incoming: object) -> object:
    if isinstance(existing, SandboxSeed):
        return existing
    if isinstance(incoming, SandboxSeed):
        return incoming
    if isinstance(existing, SandboxSpec) and isinstance(incoming, SandboxSpec):
        if existing != incoming:
            raise ValueError("Sandbox channel received multiple sandbox specs.")
        return existing
    if existing != incoming:
        raise ValueError("Sandbox channel received multiple sandbox specs.")
    return existing


def merge_sandbox_uploads(existing: object, incoming: object) -> dict[str, object]:
    uploads = sandbox_uploads(existing)
    for name, source in sandbox_uploads(incoming).items():
        if name in uploads and uploads[name] != source:
            raise ValueError(f"Duplicate sandbox upload name: {name!r}")
        uploads[name] = source
    return uploads


def sandbox_uploads(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Sandbox uploads must be a mapping.")
    return {str(name): source for name, source in value.items()}


def channel_logger(context: ChannelContext) -> logging.Logger:
    for owner in context.owners:
        logger = getattr(owner, "logger", None)
        if isinstance(logger, logging.Logger):
            return logger
    return logging.getLogger(__name__)


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return as_float(value)


def as_int(value: object) -> int:
    return int(cast(Any, value))


def as_float(value: object) -> float:
    return float(cast(Any, value))


def sandbox_timeouts(value: object) -> SandboxTimeouts:
    if value is None:
        return SandboxTimeouts()
    if isinstance(value, SandboxTimeouts):
        return value
    if isinstance(value, Mapping):
        return SandboxTimeouts(**dict(value))
    raise TypeError("Sandbox runtime timeouts must be SandboxTimeouts or a mapping.")
