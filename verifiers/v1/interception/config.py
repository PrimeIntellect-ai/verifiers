"""How the host interception server is made reachable from the harness's runtime.

The interception server runs on the host; a harness in a remote runtime (a prime/modal sandbox)
reaches it over a tunnel. `InterceptionConfig` is the discriminated union choosing the type; the
matching `Tunnel` (see `verifiers.v1.interception.tunnel`, picked by `tunnel_cls`) implements it:

- `prime` (default): `prime_tunnel` (frpc) — works from any host with prime credentials;
- `modal`: Modal's own port forwarding (`modal.forward`) — only when the framework itself runs
  inside a Modal container (a Modal-hosted trainer/eval); `modal.forward` refuses elsewhere;
- `custom`: bring your own reverse proxy — the framework opens no tunnel and trusts a public `url`
  you front the interception port with (nginx/caddy, an ngrok tunnel, ...).
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


class ModalInterceptionConfig(BaseInterceptionConfig):
    """Expose the host interception port via Modal's own forwarding (`modal.forward`)."""

    type: Literal["modal"] = "modal"


class CustomInterceptionConfig(BaseInterceptionConfig):
    """Bring your own reverse proxy: the framework opens no tunnel and reaches the interception
    server at `url`, which you front the fixed local `port` with. One public URL is one
    interception server, so every rollout shares it (`multiplex` doesn't apply)."""

    type: Literal["custom"] = "custom"
    url: str
    """Public base URL your reverse proxy serves, forwarding to the host's `port` (no trailing
    slash). The model route is `{url}/v1`; the tool/user state channels are `{url}/state` +
    `/task`."""
    port: int = Field(ge=1, le=65535)
    """Fixed local port the interception server binds, so your reverse proxy has a stable target."""

    def model_post_init(self, _ctx) -> None:
        self.url = self.url.rstrip("/")


InterceptionConfig = Annotated[
    PrimeInterceptionConfig | ModalInterceptionConfig | CustomInterceptionConfig,
    Field(discriminator="type"),
]
