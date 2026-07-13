"""The interception layer.

`server` catches a harness's model calls and proxies them to the client, multiplexing many
rollouts on one server keyed by their secret — and is itself the single-server
`Interception`. `pool` composes such servers, statically (a fixed set, e.g. bring-your-own
endpoints) or elastically (grown on demand). `tunnel` makes a server reachable from remote
consumers. `make_interception` picks the shape from the `InterceptionConfig` type.
"""

from verifiers.v1.interception.base import Interception, Slot
from verifiers.v1.interception.config import (
    BaseInterceptionConfig,
    ElasticInterceptionPoolConfig,
    InterceptionConfig,
    InterceptionServerConfig,
    StaticInterceptionPoolConfig,
)
from verifiers.v1.interception.pool import (
    ElasticInterceptionPool,
    StaticInterceptionPool,
)
from verifiers.v1.interception.server import (
    InterceptionServer,
    RolloutLimits,
    RolloutSession,
)
from verifiers.v1.interception.tunnel import make_tunnel


def make_interception(config: InterceptionConfig, *, is_local: bool) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to
    `make_runtime`). `is_local` (no consumer — harness or tool/user server — runs in a
    remote runtime) means the servers get no tunnel and are reached at localhost;
    otherwise each is exposed via its configured tunnel. The caller computes it (see
    `Environment.interception`)."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(None if is_local else make_tunnel(config.tunnel))
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(
            [
                InterceptionServer(None if is_local else make_tunnel(server.tunnel))
                for server in config.servers
            ]
        )
    return ElasticInterceptionPool(multiplex=config.multiplex, is_local=is_local)


__all__ = [
    "Interception",
    "Slot",
    "make_interception",
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
