"""The interception layer.

`server` catches an harness's chat-completions and proxies them to the model, multiplexing many
rollouts on one server keyed by their secret — and is itself the single-server `Interception` (the
custom case). `pool` composes many such servers (one tunnel each, behind a remote runtime) for the
prime case. `make_interception` picks the shape from the `InterceptionConfig` type.
"""

from verifiers.v1.interception.base import Interception
from verifiers.v1.interception.config import (
    BaseInterceptionConfig,
    CustomInterceptionConfig,
    InterceptionConfig,
    PrimeInterceptionConfig,
)
from verifiers.v1.interception.pool import InterceptionPool, PooledServer
from verifiers.v1.interception.server import (
    InterceptionServer,
    RolloutLimits,
    RolloutSession,
)
from verifiers.v1.interception.tunnel import CustomTunnel


def make_interception(is_local: bool, config: InterceptionConfig) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to `make_runtime`):
    a single `InterceptionServer` at the BYO endpoint for `custom`, else a pooled `InterceptionPool`
    for `prime`. `is_local` (no consumer — harness or tool/user server — runs in a remote runtime)
    decides whether the server is reached at localhost or exposed via a tunnel; the caller computes
    it (see `Environment.interception`)."""
    if isinstance(config, CustomInterceptionConfig):
        return InterceptionServer(CustomTunnel(config), is_local)
    return InterceptionPool(is_local, config)


__all__ = [
    "Interception",
    "make_interception",
    "InterceptionPool",
    "PooledServer",
    "InterceptionServer",
    "RolloutLimits",
    "RolloutSession",
    "BaseInterceptionConfig",
    "InterceptionConfig",
    "PrimeInterceptionConfig",
    "CustomInterceptionConfig",
]
