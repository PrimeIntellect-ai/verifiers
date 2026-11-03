"""Interception configs: how rollouts reach a host interception server, and the tunnel
that exposes it to remote consumers."""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_config import BaseConfig


class BaseTunnelConfig(BaseConfig):
    """Base for the tunnel types — the discriminated union's common type. Per-type fields
    live on the subclasses (custom's `url`/`port`)."""


class PrimeTunnelConfig(BaseTunnelConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). No fields — the tunnel
    service mints a fresh public URL per exposed port."""

    type: Literal["prime"] = "prime"


class CustomTunnelConfig(BaseTunnelConfig):
    """Bring your own endpoint: the framework opens no tunnel and reaches the interception
    server at `url`. The interception port is plaintext HTTP (auth'd by the per-rollout
    secret) — front it with a TLS proxy / firewall on an untrusted network."""

    type: Literal["custom"] = "custom"
    url: str
    """Public base URL the server is reached at (no trailing slash) — a pre-started tunnel
    or reverse proxy's URL, or `http://<host>:<port>` for a direct bind. The model route is
    `{url}/v1`; the tool-server state channels are `{url}/state` + `/task`."""
    port: int = Field(ge=1, le=65535)
    """Fixed local port the interception server binds (on all interfaces) — your tunnel or
    proxy's target, or the public port for a direct bind."""

    def model_post_init(self, _ctx) -> None:
        self.url = self.url.rstrip("/")


# Discriminated on `type` so the CLI selects with `--interception.tunnel.type prime|custom`.
TunnelConfig = Annotated[
    PrimeTunnelConfig | CustomTunnelConfig, Field(discriminator="type")
]


class BaseInterceptionConfig(BaseConfig):
    """Base for the interception types — the discriminated union's common type. Per-type
    fields live on the subclasses (server's `tunnel`, static's `servers`, elastic's
    `multiplex`)."""


class InterceptionServerConfig(BaseInterceptionConfig):
    """A single interception server shared by every rollout, reached (when any consumer is
    remote) via its `tunnel` — the shape that supports a bring-your-own endpoint
    (`tunnel.type custom`)."""

    type: Literal["server"] = "server"
    tunnel: TunnelConfig = PrimeTunnelConfig()
    """How remote consumers reach the server: `prime` (a framework-minted prime_tunnel) or
    `custom` (a pre-started tunnel / reverse proxy / direct bind you provide)."""


class StaticInterceptionPoolConfig(BaseInterceptionConfig):
    """A fixed set of interception servers, each configured like a `server` type; rollouts
    land on the least-loaded one. The shape for multiple bring-your-own endpoints (one
    `custom` tunnel per server)."""

    type: Literal["static"] = "static"
    servers: list[InterceptionServerConfig] = Field(min_length=1)
    """One entry per server, each with its own `tunnel` choice."""


class ElasticInterceptionPoolConfig(BaseInterceptionConfig):
    """An eagerly warmed interception server, then more grown on demand: `multiplex`
    rollouts share one server (and, behind a remote consumer, one prime tunnel). The default."""

    type: Literal["elastic"] = "elastic"
    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and tunnel). N concurrent rollouts use
    ~N/multiplex servers + tunnels instead of one each — key past the per-token tunnel cap.
    1 = a server (+ tunnel) per rollout."""


# Discriminated on `type` so the CLI selects with `--interception.type server|static|elastic`.
InterceptionConfig = Annotated[
    InterceptionServerConfig
    | StaticInterceptionPoolConfig
    | ElasticInterceptionPoolConfig,
    Field(discriminator="type"),
]
