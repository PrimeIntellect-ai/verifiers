from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
from verifiers.v1.runtimes import Runtime
from verifiers.v1.session import RolloutSession


def make_interception(
    config: InterceptionConfig, *, requires_tunnel: bool
) -> Interception:
    """The interception for a config, picked by type (the host-side counterpart to
    `make_backend`). With `requires_tunnel`, each server is exposed through its configured
    tunnel; otherwise it remains on host loopback. Only a remote harness needs the
    tunnel: every tool server off the evaluator sits beside a remote harness."""
    if isinstance(config, InterceptionServerConfig):
        return InterceptionServer(config, requires_tunnel)
    if isinstance(config, StaticInterceptionPoolConfig):
        return StaticInterceptionPool(config, requires_tunnel)
    return ElasticInterceptionPool(config, requires_tunnel)


@asynccontextmanager
async def serve_interception(
    interception: Interception | None,
    runtime: Runtime,
    session: RolloutSession,
) -> AsyncIterator[Slot]:
    """A slot on the shared interception when one was injected (its owner keeps the
    lifecycle), else on a per-rollout `InterceptionServer` owned — brought up and torn
    down — by the caller's context."""
    if interception is not None:
        async with interception.acquire(session) as slot:
            yield slot
        return
    server = InterceptionServer(requires_tunnel=not runtime.is_local)
    async with server, server.acquire(session) as slot:
        yield slot


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
    "serve_interception",
]
