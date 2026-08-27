import json
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.errors import HarnessError
from verifiers.v1.interception.base import BaseInterceptionConfig, Interception, Slot
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    ElasticInterceptionPoolConfig,
    StaticInterceptionPool,
    StaticInterceptionPoolConfig,
)
from verifiers.v1.interception.server import (
    InterceptionServer,
    InterceptionServerConfig,
)
from verifiers.v1.runtimes import Runtime, runtime_is_local
from verifiers.v1.session import RolloutSession

if TYPE_CHECKING:
    from verifiers.v1.mcp import SharedToolServer

# Discriminated on `type` so the CLI selects with `--interception.type server|static|elastic`.
InterceptionConfig = Annotated[
    InterceptionServerConfig
    | StaticInterceptionPoolConfig
    | ElasticInterceptionPoolConfig,
    Field(discriminator="type"),
]

DIRECT_TOOL_SOURCE = (Path(__file__).resolve().parent / "direct.py").read_text()
TOOL_CONTENT_SOURCE = (Path(__file__).resolve().parent / "content.mjs").read_text()
DIRECT_CHAT_SOURCE = (
    (Path(__file__).resolve().parent / "chat.py")
    .read_text()
    .replace("# {tool_interception}", DIRECT_TOOL_SOURCE)
)


def prepare_tool_interception(
    args: list[str],
    runtime: Runtime,
    configuration: tuple[str, str] | None,
    harness: str,
) -> bytes | None:
    """Add direct-hook transport arguments and return the stdin credential payload."""
    if configuration is None:
        return None
    if not runtime.supports_live_processes:
        raise HarnessError(
            f"{harness} tool interception requires a runtime with live process support"
        )
    url, secret = configuration
    payload = secret.encode()
    args += [
        f"--tool-interception-url={url}",
        f"--tool-interception-secret-bytes={len(payload)}",
    ]
    return payload


async def stage_tool_interception_config(
    runtime: Runtime,
    directory: str,
    url: str,
    secret: str,
) -> str:
    """Stage one private, single-use native-hook configuration."""
    path = f"{directory}/tool-interception-{uuid.uuid4().hex}.credentials"
    payload = json.dumps({"url": url, "secret": secret}).encode()
    result = await runtime.run_with_input(
        [
            "sh",
            "-c",
            'umask 077; set -C; head -c "$1" > "$2"',
            "write-tool-credentials",
            str(len(payload)),
            path,
        ],
        {},
        payload,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            "failed to stage tool interception credentials: "
            f"{result.stderr.strip()[-500:]}"
        )
    return path


def requires_tunnel(
    harness_is_local: bool,
    server_configs: Iterable[BaseConfig] = (),
    shared: "Iterable[SharedToolServer]" = (),
) -> bool:
    """Whether the interception must be exposed via a tunnel — some consumer is off the
    host network: the harness itself, a live `shared` server in a remote runtime, or a
    tool server config placing one there (each reaches the `/state` channel from
    its own runtime). Skipped as non-consumers: a `colocated` server (shares the
    harness's runtime, covered by `harness_is_local`), a config-`url` server (external —
    it connects out), and an `external` shared server (outside the state machinery
    entirely). False means every consumer reaches the server at localhost."""
    if not harness_is_local:
        return True
    if any(not s.external and not s.local for s in shared):
        return True
    for config in server_configs:
        if getattr(config, "url", None) or config.colocated:
            continue
        if not runtime_is_local(config.runtime):
            return True
    return False


def make_interception(
    config: InterceptionConfig,
    *,
    requires_tunnel: bool,
    state_service_secrets: tuple[str, ...] = (),
) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to
    `make_runtime`). With `requires_tunnel`, each server is exposed through its configured
    tunnel; otherwise it remains on host loopback. The caller computes this requirement."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(config, requires_tunnel, state_service_secrets)
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(config, requires_tunnel, state_service_secrets)
    return ElasticInterceptionPool(config, requires_tunnel, state_service_secrets)


@asynccontextmanager
async def serve_interception(
    interception: Interception | None,
    runtime: Runtime,
    session: RolloutSession,
    servers: list,
    shared_tools: "dict[str, SharedToolServer]",
) -> AsyncIterator[Slot]:
    """A slot on the shared interception when one was injected (its owner keeps the
    lifecycle), else on a per-rollout `InterceptionServer` owned — brought up and torn
    down — by the caller's context."""
    if interception is not None:
        async with interception.acquire(session) as slot:
            yield slot
        return
    tunneled = requires_tunnel(
        runtime.is_local,
        [server.config for server in servers],
        shared_tools.values(),
    )
    server = InterceptionServer(
        requires_tunnel=tunneled,
        state_service_secrets=tuple(
            tool.state_secret for tool in shared_tools.values() if tool.state_secret
        ),
    )
    async with server, server.acquire(session) as slot:
        yield slot


__all__ = [
    "DIRECT_CHAT_SOURCE",
    "DIRECT_TOOL_SOURCE",
    "TOOL_CONTENT_SOURCE",
    "BaseInterceptionConfig",
    "ElasticInterceptionPool",
    "ElasticInterceptionPoolConfig",
    "Interception",
    "InterceptionConfig",
    "InterceptionServer",
    "InterceptionServerConfig",
    "Slot",
    "StaticInterceptionPool",
    "StaticInterceptionPoolConfig",
    "make_interception",
    "prepare_tool_interception",
    "requires_tunnel",
    "serve_interception",
    "stage_tool_interception_config",
]
