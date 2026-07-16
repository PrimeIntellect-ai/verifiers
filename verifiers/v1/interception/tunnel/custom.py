"""Custom tunnel: bring your own endpoint — open no tunnel, reach the server at the
configured `url`. The server binds all interfaces (`0.0.0.0`) on the fixed local `port`, so
`url` can be a pre-started tunnel or reverse proxy you front it with, or a direct
`http://<host>:<port>` on a reachable host."""

import contextlib
from collections.abc import AsyncIterator
from typing import ClassVar, Literal

from pydantic import Field

from verifiers.v1.interception.tunnel.base import BaseTunnelConfig, Tunnel


class CustomTunnelConfig(BaseTunnelConfig):
    """Bring your own endpoint: the framework opens no tunnel and reaches the interception
    server at `url`. The interception port is plaintext HTTP (auth'd by the per-rollout
    secret) — front it with a TLS proxy / firewall on an untrusted network."""

    type: Literal["custom"] = "custom"
    url: str
    """Public base URL the server is reached at (no trailing slash) — a pre-started tunnel
    or reverse proxy's URL, or `http://<host>:<port>` for a direct bind. The model route is
    `{url}/v1`; the tool/user state channels are `{url}/state` + `/task`."""
    port: int = Field(ge=1, le=65535)
    """Fixed local port the interception server binds (on all interfaces) — your tunnel or
    proxy's target, or the public port for a direct bind."""

    def model_post_init(self, _ctx) -> None:
        self.url = self.url.rstrip("/")


class CustomTunnel(Tunnel[CustomTunnelConfig]):
    bind_host: ClassVar[str] = "0.0.0.0"

    @property
    def bind_port(self) -> int:
        return self.config.port

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        yield self.config.url
