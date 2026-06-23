"""Prime tunnel: expose the host interception port via prime_tunnel (frpc) — the shared host
endpoint. The default; works from any host with prime credentials, for harnesses in prime *or*
modal sandboxes alike.

`host_endpoint` (and its retry helper `open_tunnel`) is the host-side prime_tunnel bridge — used
both by `PrimeTunnel` here and by the tool-serving `reachable_url` path — so both go through the
one host-wide prime_tunnel rate limit (`TUNNEL_LIMITER`, a shared resource that stays with the
other host limiters in `runtimes.limiters`).
"""

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

from verifiers.v1.interception.tunnel.base import Tunnel
from verifiers.v1.retries import retrying

TunnelT = TypeVar("TunnelT")


async def open_tunnel(
    start: Callable[[], Awaitable[TunnelT]], what: str, *, retries: int = 3
) -> TunnelT:
    """Open a host tunnel via `start`, retrying transient failures and raising `TunnelError` if it
    still fails. Tunnel creation is network-bound and globally rate-capped (`prime_tunnel` —
    512/min), so a transient failure is common and worth a few retries. `what` names it in the error."""
    from verifiers.v1.errors import TunnelError

    try:
        async for attempt in retrying(retries=retries, label=what):
            with attempt:
                return await start()
    except Exception as e:
        raise TunnelError(f"{what} failed after {retries} retries: {e}") from e
    raise TunnelError(f"{what} failed")  # unreachable: retrying() returns or reraises


@contextlib.asynccontextmanager
async def host_endpoint(port: int, is_local: bool, labels: list[str] | None = None):
    """Yield a URL a program *inside a runtime* uses to reach a HOST service on `port`. A local
    runtime shares the host network → localhost; a remote one needs a host-side reverse tunnel
    (`prime_tunnel`), torn down on exit. Shared by `PrimeTunnel` and the tool-serving `reachable_url`
    path, so both go through the same host-wide tunnel rate limit."""
    if is_local:
        yield f"http://127.0.0.1:{port}"
        return
    from prime_tunnel import Tunnel as PrimeTunnelClient

    from verifiers.v1.runtimes.limiters import TUNNEL_LIMITER

    async def _start():
        tunnel = PrimeTunnelClient(local_port=port, labels=labels or None)
        async with TUNNEL_LIMITER:  # shared prime_tunnel rate (512/min, host-wide)
            return tunnel, str(await tunnel.start()).rstrip("/")

    tunnel, url = await open_tunnel(_start, f"host tunnel (port {port})")
    try:
        yield url
    finally:
        with contextlib.suppress(Exception):
            tunnel.sync_stop()


class PrimeTunnel(Tunnel):
    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        async with host_endpoint(port, is_local=False) as url:
            yield url
