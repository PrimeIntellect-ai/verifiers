"""How the host interception server is made reachable from the harness's runtime.

The interception server runs on the host; a harness in a remote runtime (a prime/modal sandbox)
reaches it over a tunnel. `InterceptionConfig` is the discriminated union choosing the type; the
matching `Tunnel` (see `verifiers.v1.interception.tunnel`) implements it:

- `prime` (default): `prime_tunnel` (frpc) — works from any host with prime credentials;
- `custom`: bring your own endpoint — the framework opens no tunnel and trusts a public `url`. The
  server binds all interfaces on a fixed `port`; `url` is either a reverse proxy you front it with or
  a direct `http://<host>:<port>` on a host the harness can reach.
"""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_config import BaseConfig


class BaseInterceptionConfig(BaseConfig):
    """Base for the interception types — the discriminated union's common type. Per-type fields live
    on the subclasses (prime's `multiplex`, custom's `url`/`port`)."""


class PrimeInterceptionConfig(BaseInterceptionConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). The default. Pooled: `multiplex`
    rollouts share one server (one tunnel), grown on demand."""

    type: Literal["prime"] = "prime"
    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and tunnel). N concurrent rollouts use
    ~N/multiplex servers + tunnels instead of one each — key past the per-token tunnel cap.
    1 = a server (+ tunnel) per rollout."""


class CustomInterceptionConfig(BaseInterceptionConfig):
    """Bring your own endpoint: the framework opens no tunnel and reaches the interception server at
    `url`. The server binds all interfaces on the fixed local `port`, so `url` is either a reverse
    proxy you front it with or a direct `http://<host>:<port>` on a reachable host. One URL is one
    server, so every rollout shares it — no pool, no multiplex. The interception port is plaintext
    HTTP (auth'd by the per-rollout secret) — front it with a TLS proxy / firewall on an untrusted
    network."""

    type: Literal["custom"] = "custom"
    url: str
    """Public base URL the harness reaches the interception server at (no trailing slash) — a reverse
    proxy's URL, or `http://<host>:<port>` for a direct bind. The model route is `{url}/v1`; the
    tool/user state channels are `{url}/state` + `/task`."""
    port: int = Field(ge=1, le=65535)
    """Fixed local port the interception server binds (on all interfaces) — your reverse proxy's
    target, or the public port for a direct bind."""

    def model_post_init(self, _ctx) -> None:
        self.url = self.url.rstrip("/")


InterceptionConfig = Annotated[
    PrimeInterceptionConfig | CustomInterceptionConfig,
    Field(discriminator="type"),
]
