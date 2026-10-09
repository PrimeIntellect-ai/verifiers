"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in prime *or* modal sandboxes
alike — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from verifiers.v1.interception.tunnel.base import BaseTunnelConfig, Tunnel
from verifiers.v1.runtimes.limiters import CreationLimiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.scope import run_scope

if TYPE_CHECKING:
    from prime_tunnel import Tunnel as TunnelClient

logger = logging.getLogger(__name__)


class TunnelLogHandler(logging.Handler):
    """Forward tunnel warnings and errors to this module's logger — and, when the app
    enables INFO on `prime_tunnel`, the login after a reconnect."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().removeprefix("frpc ")
        if record.levelno >= logging.WARNING or "login to server success" in message:
            logger.log(record.levelno, "tunnel %s", message)


logging.getLogger("prime_tunnel.frpc").addHandler(TunnelLogHandler())

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process of a run that opens one. One run-scoped
# limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512


def tunnel_limiter() -> CreationLimiter:
    return CreationLimiter("prime-tunnel", run_scope(), _TUNNELS_PER_MIN / 60)


# How often a held tunnel's state is checked with the tunnel service. The service marks a
# tunnel disconnected about a minute after its frpc stops answering; two such checks in a
# row restart frpc.
CHECK_SECONDS = 20


async def _watch(client: "TunnelClient", url: str) -> None:
    """Keep a held tunnel up. A tunnel service hiccup can leave frpc running while the
    service has it disconnected for good, and every request to the URL then hangs or
    404s; frpc can also exit. Either way, restart it."""
    strikes = 0
    while True:
        await asyncio.sleep(CHECK_SECONDS * random.uniform(0.75, 1.25))
        if client.is_running:
            try:
                info = await client._client.get_tunnel(client.tunnel_id)
            except Exception as e:  # noqa: BLE001 - the service is unreachable: no verdict
                logger.debug("tunnel %s: state check failed: %s", url, e)
                continue
            if info is None:
                logger.error("tunnel %s: registration gone; cannot repair", url)
                return
            strikes = strikes + 1 if info.status == "disconnected" else 0
            if strikes < 2:
                continue
        logger.warning(
            "tunnel %s: %s; restarting frpc",
            url,
            "disconnected" if client.is_running else "frpc exited",
        )
        try:
            await client.restart()
        except Exception as e:  # noqa: BLE001 - tried again on the next check
            logger.warning("tunnel %s: frpc restart failed: %s", url, e)
        else:
            logger.warning("tunnel %s: back up", url)
            strikes = 0


class PrimeTunnelConfig(BaseTunnelConfig):
    """Expose the host interception port via `prime_tunnel` (frpc). No fields — the tunnel
    service mints a fresh public URL per exposed port."""

    type: Literal["prime"] = "prime"


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    def __init__(self, config: PrimeTunnelConfig | None = None) -> None:
        ensure_prime_auth()
        super().__init__(config)

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and rate-capped (512/min, run-wide via the shared
        `tunnel_limiter`), so transient failures are retried; a terminal one raises
        `TunnelError`. While held, frpc is restarted if the tunnel goes down. The tunnel is
        torn down on exit."""
        from prime_tunnel import Tunnel as TunnelClient

        from verifiers.v1.errors import TunnelError
        from verifiers.v1.utils.retries import retrying

        label = f"host tunnel (port {port})"
        try:
            async for attempt in retrying(retries=3, label=label):
                with attempt:
                    client = TunnelClient(local_port=port)
                    async with tunnel_limiter():
                        url = str(await client.start()).rstrip("/")
        except Exception as e:
            raise TunnelError(f"{label} failed: {e}") from e
        watch = asyncio.create_task(_watch(client, url))

        async def close() -> None:
            watch.cancel()  # first, so it can't start a new frpc after the stop
            await asyncio.wait({watch})
            await asyncio.to_thread(client.sync_stop)

        try:
            yield url
        finally:
            # Run the stop to completion even under cancellation (`run_shielded` re-raises
            # the cancellation after); tunnel-stop failures are best-effort.
            with contextlib.suppress(Exception):
                await run_shielded(close())
