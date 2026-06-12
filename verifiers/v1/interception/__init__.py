"""The interception layer.

`server` catches an harness's chat-completions and proxies them to the model, multiplexing
many rollouts on one server keyed by their secret; `pool` shares those servers (one tunnel
each, behind a remote runtime) across all of an eval's or env-server's rollouts.
"""

from verifiers.v1.interception.pool import InterceptionPool, PooledServer
from verifiers.v1.interception.server import (
    InterceptionServer,
    RolloutLimits,
    RolloutSession,
)

__all__ = [
    "InterceptionPool",
    "PooledServer",
    "InterceptionServer",
    "RolloutLimits",
    "RolloutSession",
]
