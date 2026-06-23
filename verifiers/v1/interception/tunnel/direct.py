"""Direct tunnel: bind the interception server directly on a reachable host interface — no tunnel,
no proxy. Each server binds an ephemeral port on `config.bind_host` (0.0.0.0 by default) and is
reached at `http://{config.host}:{port}`; the pool multiplexes across servers like prime (no tunnel
rate cap here, so one server already serves many rollouts — spreading is optional). Plaintext HTTP
carrying the per-rollout secret, so for a trusted network only."""

import contextlib
from collections.abc import AsyncIterator

from verifiers.v1.interception.config import DirectInterceptionConfig
from verifiers.v1.interception.tunnel.base import Tunnel


class DirectTunnel(Tunnel[DirectInterceptionConfig]):
    @property
    def bind_host(self) -> str:
        return self.config.bind_host

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        yield f"http://{self.config.host}:{port}"
