"""Direct tunnel: bind the interception server directly on a reachable host interface — no tunnel,
no proxy. Each server binds an ephemeral port on `config.bind_host` (defaults to `config.host`, so it
listens only on the advertised interface) and is reached at `http://{config.host}:{port}`; the pool
multiplexes across servers like prime (no tunnel rate cap here, so one server already serves many
rollouts — spreading is optional). Plaintext HTTP carrying the per-rollout secret, so for a trusted
network only."""

import contextlib
from collections.abc import AsyncIterator

from verifiers.v1.interception.config import DirectInterceptionConfig
from verifiers.v1.interception.tunnel.base import Tunnel


class DirectTunnel(Tunnel[DirectInterceptionConfig]):
    @property
    def bind_host(self) -> str:
        # listen only on the advertised interface by default (not all of them)
        return self.config.bind_host or self.config.host

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        yield f"http://{self.config.host}:{port}"
