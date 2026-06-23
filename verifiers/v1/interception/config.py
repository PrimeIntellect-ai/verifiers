"""How the host interception server is made reachable from the harness's runtime.

The interception server runs on the host; a harness in a remote runtime (a prime/modal sandbox)
reaches it over a tunnel. `InterceptionConfig` is the discriminated union choosing the type; the
matching `Tunnel` (see `verifiers.v1.interception.tunnel`, picked by `tunnel_cls`) implements it:

- `prime` (default): `prime_tunnel` (frpc) — works from any host with prime credentials;
- `custom`: bring your own endpoint — the framework opens no tunnel and trusts a public `url`. Front
  the loopback `port` with a reverse proxy (default), or set `bind_host` to expose the `port`
  directly (`url=http://<host>:<port>`, no proxy) on a host the harness can reach.
"""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_config import BaseConfig


class BaseInterceptionConfig(BaseConfig):
    """Fields shared by every interception type."""

    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and, behind a remote runtime, one tunnel).
    N concurrent rollouts use ~N/multiplex servers + tunnels instead of one each — key past the
    per-token tunnel cap. 1 = a server (+ tunnel) per rollout. `custom` ignores it (one BYO
    endpoint is structurally a single server)."""


class PrimeInterceptionConfig(BaseInterceptionConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). The default."""

    type: Literal["prime"] = "prime"


class CustomInterceptionConfig(BaseInterceptionConfig):
    """Bring your own endpoint: the framework opens no tunnel and reaches the interception server at
    `url`. By default it binds the fixed local `port` on loopback for a reverse proxy you front it
    with; set `bind_host` to a reachable interface to expose the `port` directly (no proxy), with
    `url=http://<host>:<port>`. One URL is one server, so every rollout shares it (`multiplex`
    doesn't apply)."""

    type: Literal["custom"] = "custom"
    url: str
    """Public base URL the harness reaches the interception server at (no trailing slash). The model
    route is `{url}/v1`; the tool/user state channels are `{url}/state` + `/task`."""
    port: int = Field(ge=1, le=65535)
    """Fixed local port the interception server binds — your reverse proxy's target, or the public
    port for a direct bind."""
    bind_host: str = "127.0.0.1"
    """Interface the server listens on. Loopback (default) for a same-host reverse proxy; set a
    reachable interface (`0.0.0.0`, or a specific public/LAN IP) to expose `port` directly with no
    proxy. A direct bind is plaintext HTTP carrying the per-rollout secret — trusted-network only."""

    def model_post_init(self, _ctx) -> None:
        self.url = self.url.rstrip("/")


InterceptionConfig = Annotated[
    PrimeInterceptionConfig | CustomInterceptionConfig,
    Field(discriminator="type"),
]
