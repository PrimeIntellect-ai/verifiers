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
from verifiers.v1.runtimes import RuntimeConfig, runtime_is_local


def make_interception(
    runtime_config: RuntimeConfig, config: InterceptionConfig
) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to `make_runtime`):
    a single `InterceptionServer` at the BYO endpoint for `custom`, else a pooled `InterceptionPool`
    for `prime`. `runtime_config` is the harness runtime, whose locality decides reachability."""
    if isinstance(config, CustomInterceptionConfig):
        return InterceptionServer(CustomTunnel(config), runtime_is_local(runtime_config))
    return InterceptionPool(runtime_config, config)


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
