"""Custom tunnel: bring your own endpoint — open no tunnel, reach the server at the
configured `url`. The server binds all interfaces (`0.0.0.0`) on the fixed local `port`, so
`url` can be a pre-started tunnel or reverse proxy you front it with, or a direct
`http://<host>:<port>` on a reachable host."""

import contextlib
from collections.abc import AsyncIterator
from typing import ClassVar

from verifiers.v1.configs.interception import CustomTunnelConfig
from verifiers.v1.interception.tunnel.base import Tunnel


class CustomTunnel(Tunnel[CustomTunnelConfig]):
    bind_host: ClassVar[str] = "0.0.0.0"

    @property
    def bind_port(self) -> int:
        return self.config.port

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        yield self.config.url
