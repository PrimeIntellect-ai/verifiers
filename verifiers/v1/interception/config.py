"""The interception shapes, as config: one server, a static pool, or an elastic pool.

`InterceptionConfig` is the discriminated union picking the `Interception` a run gets
(see `make_interception`):

- `server`: a single `InterceptionServer` every rollout shares, with its own `tunnel`
  choice (prime, or a bring-your-own `custom` endpoint);
- `static`: a fixed list of such servers (`StaticInterceptionPool`), least-loaded — the
  shape for pre-started endpoints, one per server;
- `elastic` (default): servers grown on demand (`ElasticInterceptionPool`), `multiplex`
  rollouts per server. Always prime tunnels — the only kind the framework can mint while
  scaling — so there's no tunnel choice here.
"""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.interception.tunnel import PrimeTunnelConfig, TunnelConfig


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
    """Interception servers grown on demand: `multiplex` rollouts share one server (and,
    behind a remote consumer, one prime tunnel). The default."""

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
