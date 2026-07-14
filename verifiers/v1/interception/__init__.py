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
)
from verifiers.v1.runtimes import runtime_reaches_host_locally

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
    """Whether interception needs a public tunnel because any consumer is remote.

    Local runtimes map framework routes through their network policy. Colocated servers
    share that route; configured URLs and external shared servers do not consume state."""
    if not harness_is_local:
        return True
    if any(not s.external and not s.reaches_host for s in shared):
        return True
    for config in server_configs:
        if getattr(config, "url", None) or config.colocated:
            continue
        if not runtime_reaches_host_locally(config.runtime):
            return True
    return False


def make_interception(
    config: InterceptionConfig,
    *,
    requires_tunnel: bool,
    extra_host: str | None = None,
) -> Interception:
    """Build the configured interception shape for the resolved network topology."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(config, requires_tunnel, extra_host=extra_host)
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(config, requires_tunnel, extra_host=extra_host)
    return ElasticInterceptionPool(config, requires_tunnel, extra_host=extra_host)


__all__ = [
    "Interception",
    "Slot",
    "make_interception",
    "requires_tunnel",
    "InterceptionServer",
    "StaticInterceptionPool",
    "ElasticInterceptionPool",
    "BaseInterceptionConfig",
    "InterceptionConfig",
    "InterceptionServerConfig",
    "StaticInterceptionPoolConfig",
    "ElasticInterceptionPoolConfig",
]
