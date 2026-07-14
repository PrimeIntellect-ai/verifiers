"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in prime *or* modal sandboxes
alike — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Literal

from verifiers.v1.interception.tunnel.base import BaseTunnelConfig, Tunnel
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process on the host that opens one. One host-global
# limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512
TUNNEL_LIMITER = creation_limiter(_TUNNELS_PER_MIN / 60, "prime-tunnel")


class PrimeTunnelConfig(BaseTunnelConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). No fields — the tunnel
    service mints a fresh public URL per exposed port."""

    type: Literal["prime"] = "prime"


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and globally rate-capped (512/min, host-wide via the shared
        `TUNNEL_LIMITER`), so transient failures are retried; a terminal one raises
        `TunnelError`. The tunnel is torn down on exit."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.retries import retrying

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
