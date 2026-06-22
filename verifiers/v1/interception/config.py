"""How the host interception server is made reachable from the harness's runtime.

The interception server runs on the host; a harness in a remote runtime (a prime/modal sandbox)
reaches it over a tunnel. `InterceptionConfig` is the discriminated union choosing how that
tunnel is built:

- `prime` (default): `prime_tunnel` (frpc) — works from any host with prime credentials;
- `modal`: Modal's own port forwarding (`modal.forward`) — only when the framework itself runs
  inside a Modal container (a Modal-hosted trainer/eval); `modal.forward` refuses elsewhere;
- `url`: bring your own reverse proxy — the framework opens no tunnel and trusts a public `url`
  you front the interception port with (nginx/caddy, an ngrok tunnel, ...).

`multiplex` is the field shared by all three: how many rollouts share one interception server
(and, behind a remote runtime, one tunnel). It moved off `EnvConfig` onto here, since it sizes
the same tunnel fan-out the type chooses.
"""

import contextlib
from typing import Annotated, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import host_endpoint, open_tunnel


class BaseInterceptionConfig(BaseConfig):
    """Fields shared by every interception type."""

    multiplex: int = Field(32, ge=1)
    """Rollouts that share one interception server (and, behind a remote runtime, one tunnel).
    N concurrent rollouts use ~N/multiplex servers + tunnels instead of one each — key past the
    per-token tunnel cap. 1 = a server (+ tunnel) per rollout. `url` ignores it (one BYO endpoint
    is structurally a single server)."""


class PrimeInterceptionConfig(BaseInterceptionConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). The default — works from any
    host with prime credentials, for harnesses in prime *or* modal sandboxes alike."""

    type: Literal["prime"] = "prime"


class ModalInterceptionConfig(BaseInterceptionConfig):
    """Expose the host interception port via Modal's own forwarding (`modal.forward`). Only works
    when the framework itself runs inside a Modal container (e.g. a Modal-hosted trainer/eval);
    `modal.forward` raises anywhere else."""

    type: Literal["modal"] = "modal"


class UrlInterceptionConfig(BaseInterceptionConfig):
    """Bring your own reverse proxy: the framework opens no tunnel and reaches the interception
    server at `url`, which you front the fixed local `port` with. One public URL is one
    interception server, so every rollout shares it (`multiplex` doesn't apply)."""

    type: Literal["url"] = "url"
    url: str
    """Public base URL your reverse proxy serves, forwarding to the host's `port` (no trailing
    slash). The model route is `{url}/v1`; the tool/user state channels are `{url}/state` +
    `/task`."""
    port: int = Field(ge=1, le=65535)
    """Fixed local port the interception server binds, so your reverse proxy has a stable target."""

    def model_post_init(self, _ctx) -> None:
        self.url = self.url.rstrip("/")


InterceptionConfig = Annotated[
    PrimeInterceptionConfig | ModalInterceptionConfig | UrlInterceptionConfig,
    Field(discriminator="type"),
]


def bind_host(config: BaseInterceptionConfig) -> str:
    """The address the interception server must listen on for this type. Modal's port forwarding
    reaches the container's routable interface (not its loopback), so a modal-exposed server must
    bind `0.0.0.0`; prime (frpc) and url (a BYO proxy) reach it over loopback on the same host, so
    they keep `127.0.0.1`. Safe for modal: that server runs inside an isolated container whose only
    ingress is the `modal.forward` tunnel itself."""
    return "0.0.0.0" if isinstance(config, ModalInterceptionConfig) else "127.0.0.1"


@contextlib.asynccontextmanager
async def _modal_host_endpoint(port: int):
    """Yield a public URL forwarding to the host's `port` via Modal's own port forwarding. Requires
    the framework to run inside a Modal container — `modal.forward` raises `InvalidError` otherwise.
    The reverse of `ModalRuntime.expose` (which publishes a port *inside* a sandbox)."""
    try:
        import modal
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "modal interception requires the Modal SDK; install `verifiers[modal]`."
        ) from e
    stack = contextlib.AsyncExitStack()

    async def _start() -> str:
        tunnel = await stack.enter_async_context(modal.forward(port))
        return str(tunnel.url).rstrip("/")

    try:
        yield await open_tunnel(_start, f"modal host tunnel (port {port})")
    finally:
        await stack.aclose()


@contextlib.asynccontextmanager
async def expose_interception(config: BaseInterceptionConfig, port: int, *, is_local: bool):
    """Yield a URL the harness's runtime uses to reach the host interception server on `port`,
    built per the configured interception `type`. A local runtime shares the host network, so it's
    always reached at localhost (no tunnel, whatever the type); a remote one is bridged by the
    chosen backend. The single place the interception type maps to a reachable URL."""
    if is_local:
        yield f"http://127.0.0.1:{port}"
    elif isinstance(config, UrlInterceptionConfig):
        # BYO reverse proxy: the user already fronts `port` at `url`; we open no tunnel.
        yield config.url
    elif isinstance(config, ModalInterceptionConfig):
        async with _modal_host_endpoint(port) as url:
            yield url
    else:  # prime (default): the shared prime_tunnel host endpoint
        async with host_endpoint(port, is_local=False) as url:
            yield url
