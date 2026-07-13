"""The interception layer.

`server` catches a harness's model calls and proxies them to the client, multiplexing many
rollouts on one server keyed by their secret — and is itself the single-server
`Interception`. `pool` composes such servers, statically (a fixed set, e.g. bring-your-own
endpoints) or elastically (grown on demand). `tunnel` makes a server reachable from remote
consumers; `requires_tunnel` decides whether that's needed. `make_interception` picks the
shape from the `InterceptionConfig` type.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated

from pydantic import Field
from pydantic_config import BaseConfig

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
    RolloutLimits,
    RolloutSession,
)
from verifiers.v1.interception.tunnel import make_tunnel
from verifiers.v1.runtimes import runtime_is_local

if TYPE_CHECKING:
    from verifiers.v1.mcp import SharedToolServer

# Discriminated on `type` so the CLI selects with `--interception.type server|static|elastic`.
InterceptionConfig = Annotated[
    InterceptionServerConfig
    | StaticInterceptionPoolConfig
    | ElasticInterceptionPoolConfig,
    Field(discriminator="type"),
]


def requires_tunnel(
    harness_is_local: bool,
    server_configs: Iterable[BaseConfig] = (),
    shared: "Iterable[SharedToolServer]" = (),
) -> bool:
    """Whether the interception must be exposed via a tunnel — some consumer is off the
    host network: the harness itself, a live `shared` server in a remote runtime, or a
    tool/user server config placing one there (each reaches the `/state` channel from
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
    config: InterceptionConfig, *, requires_tunnel: bool
) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to
    `make_runtime`). With `requires_tunnel` (some consumer is off the host network — see
    the `requires_tunnel` util) each server is exposed via its configured tunnel; without
    it they get none and are reached at localhost. The caller computes it (see
    `Environment.interception`)."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(
            make_tunnel(config.tunnel) if requires_tunnel else None
        )
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(
            [
                InterceptionServer(
                    make_tunnel(server.tunnel) if requires_tunnel else None
                )
                for server in config.servers
            ]
        )
    return ElasticInterceptionPool(
        multiplex=config.multiplex, requires_tunnel=requires_tunnel
    )


__all__ = [
    "Interception",
    "Slot",
    "make_interception",
    "requires_tunnel",
    "InterceptionServer",
    "StaticInterceptionPool",
    "ElasticInterceptionPool",
    "RolloutLimits",
    "RolloutSession",
    "BaseInterceptionConfig",
    "InterceptionConfig",
    "InterceptionServerConfig",
    "StaticInterceptionPoolConfig",
    "ElasticInterceptionPoolConfig",
]
