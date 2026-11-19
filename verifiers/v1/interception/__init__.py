from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from pydantic_config import BaseConfig

from verifiers.v1.configs.interception import (
    BaseInterceptionConfig,
    ElasticInterceptionPoolConfig,
    InterceptionConfig,
    InterceptionServerConfig,
    StaticInterceptionPoolConfig,
)
from verifiers.v1.interception.base import Interception, Slot
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    StaticInterceptionPool,
)
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.runtimes import Runtime, runtime_is_local
from verifiers.v1.session import RolloutSession

if TYPE_CHECKING:
    from verifiers.v1.mcp import SharedToolServer


def requires_tunnel(
    harness_is_local: bool,
    server_configs: Iterable[BaseConfig] = (),
    shared: "Iterable[SharedToolServer]" = (),
) -> bool:
    """Whether the interception must be exposed via a tunnel — some consumer is off the
    host network: the harness itself, a live `shared` server in a remote runtime, or a
    tool server config placing one there (each reaches the `/state` channel from
    its own runtime). Task-format adapters share the harness's locality. Skipped as
    non-consumers: a config-`url` server (external —
    it connects out), and an `external` shared server (outside the state machinery
    entirely). False means every consumer reaches the server at localhost."""
    from verifiers.v1.mcp.toolset import BaseToolsetConfig

    if not harness_is_local:
        return True
    if any(not s.external and not s.local for s in shared):
        return True
    return any(
        isinstance(config, BaseToolsetConfig)
        and not config.url
        and not runtime_is_local(config.runtime)
        for config in server_configs
    )


def make_interception(
    config: InterceptionConfig,
    *,
    requires_tunnel: bool,
    state_service_secrets: tuple[str, ...] = (),
) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to
    `make_backend`). With `requires_tunnel`, each server is exposed through its configured
    tunnel; otherwise it remains on host loopback. The caller computes this requirement."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(config, requires_tunnel, state_service_secrets)
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(config, requires_tunnel, state_service_secrets)
    return ElasticInterceptionPool(config, requires_tunnel, state_service_secrets)


def _usable_interception(
    interception: Interception | None, runtime: Runtime
) -> Interception | None:
    if (
        isinstance(interception, InterceptionServer)
        and interception.lan_host
        and (runtime.type != "prime" or not runtime._remote_tools_only)
    ):
        return None
    return interception


async def start_rollout_interception(
    runtime_is_local: bool,
    toolsets: list,
    shared_tools: "dict[str, SharedToolServer]",
) -> InterceptionServer:
    """Start an owned per-rollout `InterceptionServer`."""
    server = InterceptionServer(
        requires_tunnel=requires_tunnel(
            runtime_is_local,
            [server.config for server in toolsets],
            shared_tools.values(),
        ),
        state_service_secrets=tuple(
            tool.state_secret for tool in shared_tools.values() if tool.state_secret
        ),
    )
    await server.__aenter__()
    return server


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
    session.state_service_secrets = frozenset(
        tool.state_secret for tool in shared_tools.values() if tool.state_secret
    )
    interception = _usable_interception(interception, runtime)
    if isinstance(interception, InterceptionServer) and interception.lan_host:
        from verifiers.v1.interception.tunnel import PrimeTunnel

        async with AsyncExitStack() as stack:
            session.state_base_url = f"http://127.0.0.1:{interception.port}"
            if requires_tunnel(
                True, [s.config for s in servers], shared_tools.values()
            ) or any(getattr(s.config, "service", None) for s in servers):
                session.state_base_url = await stack.enter_async_context(
                    PrimeTunnel().expose(interception.port)
                )
            yield await stack.enter_async_context(interception.acquire(session))
        return
    if interception is not None:
        async with interception.acquire(session) as slot:
            yield slot
        return
    server = await start_rollout_interception(runtime.is_local, servers, shared_tools)
    try:
        async with server.acquire(session) as slot:
            yield slot
    finally:
        await server.__aexit__(None, None, None)


__all__ = [
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
    "requires_tunnel",
    "serve_interception",
]
