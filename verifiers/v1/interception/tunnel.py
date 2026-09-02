"""Expose a host interception port to a remote consumer via prime_tunnel (frpc).

The interception server runs on the host. Local runtimes reach it at a host-local URL
directly or through their runtime translation; remote runtimes need the port published
outward. `PrimeTunnel` does that — it works from any host with prime credentials, for
consumers in prime *or* modal sandboxes alike — and is the host-side counterpart to
`Runtime.expose`, which publishes a port inside a sandbox.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator

from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process for the user that opens one. One user-global
# limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512
TUNNEL_LIMITER = creation_limiter(_TUNNELS_PER_MIN / 60, "prime-tunnel")


class PrimeTunnel:
    """Exposes a host interception port via `prime_tunnel` (frpc); the tunnel service mints
    a fresh public URL per exposed port."""

    def __init__(self) -> None:
        ensure_prime_auth()

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and globally rate-capped (512/min, user-wide via the shared
        `TUNNEL_LIMITER`), so transient failures are retried; a terminal one raises
        `TunnelError`. The tunnel is torn down on exit."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.utils.retries import retrying

        label = f"host tunnel (port {port})"
        try:
            async for attempt in retrying(retries=3, label=label):
                with attempt:
                    client = TunnelClient(local_port=port)
                    async with TUNNEL_LIMITER:
                        url = str(await client.start()).rstrip("/")
        except Exception as e:
            raise TunnelError(f"{label} failed: {e}") from e
        try:
            yield url
        finally:
            # Run the synchronous stop to completion even under cancellation (`run_shielded`
            # re-raises the cancellation after); tunnel-stop failures are best-effort.
            with contextlib.suppress(Exception):
                await run_shielded(asyncio.to_thread(client.sync_stop))


__all__ = ["TUNNEL_LIMITER", "PrimeTunnel"]
