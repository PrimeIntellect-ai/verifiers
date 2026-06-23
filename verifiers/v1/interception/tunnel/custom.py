"""Custom tunnel: bring your own reverse proxy — open no tunnel, reach the server at the configured
`url` (which you front the fixed local `port` with). One URL is one server, shared by every
rollout."""

import contextlib
from collections.abc import AsyncIterator
from typing import ClassVar

from verifiers.v1.interception.config import CustomInterceptionConfig
from verifiers.v1.interception.tunnel.base import Tunnel


class CustomTunnel(Tunnel[CustomInterceptionConfig]):
    single_server: ClassVar[bool] = True

    @property
    def bind_host(self) -> str:
        return self.config.bind_host

    @property
    def bind_port(self) -> int:
        return self.config.port

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        yield self.config.url
