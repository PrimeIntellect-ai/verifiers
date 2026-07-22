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
    """Whether interception needs a public tunnel because some consumer cannot use a
    host-local URL: the harness itself, a live `shared` server in a remote runtime, or a
    tool/user server placed there. Colocated servers, configured URLs, and external shared
    servers add no consumer. False means local URL translation is sufficient."""
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
    `make_runtime`). With `requires_tunnel`, each server is exposed through its configured
    tunnel; otherwise it remains on host loopback. The caller computes this requirement."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(config, requires_tunnel)
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(config, requires_tunnel)
    return ElasticInterceptionPool(config, requires_tunnel)


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
