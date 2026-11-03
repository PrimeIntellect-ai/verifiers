"""Prime tunnel: expose the host interception port via prime_tunnel (frpc). The default;
works from any host with prime credentials, for consumers in any remote sandbox — and the only tunnel the framework can mint on demand, so it's what the elastic
pool scales with."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from verifiers.v1.configs.interception import PrimeTunnelConfig
from verifiers.v1.interception.tunnel.base import HEALTH_PATH, Tunnel
from verifiers.v1.runtimes.limiters import CreationLimiter
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.log_scope import get_logger
from verifiers.v1.utils.prime import ensure_prime_auth
from verifiers.v1.utils.scope import run_scope

if TYPE_CHECKING:
    from prime_tunnel import Tunnel as TunnelClient

logger = get_logger(__name__)


class TunnelLogHandler(logging.Handler):
    """Forward tunnel warnings and errors to this module's logger — and, when the app
    enables INFO on `prime_tunnel`, the login after a reconnect."""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().removeprefix("frpc ").removeprefix("Tunnel ")
        if record.levelno >= logging.WARNING or "login to server success" in message:
            logger.log(record.levelno, "tunnel %s", message)


logging.getLogger("prime_tunnel").addHandler(TunnelLogHandler())

# The prime_tunnel service caps tunnel starts at 512/min per API token — a property of the
# tunnel service, shared by every process of a run that opens one. One run-scoped
# limiter, not a per-runtime config knob.
_TUNNELS_PER_MIN = 512


def tunnel_limiter() -> CreationLimiter:
    return CreationLimiter("prime-tunnel", run_scope(), _TUNNELS_PER_MIN / 60)


async def _keep_serving(client: "TunnelClient", on_gone=None) -> None:
    """Restart frpc whenever requests stop reaching the service through the URL, until the
    registration can't be used any more. The SDK logs its restarts."""
    from prime_tunnel import TunnelGoneError

    try:
        await client.keep_serving(HEALTH_PATH)
    except TunnelGoneError:
        if on_gone is not None:
            on_gone()


class PrimeTunnel(Tunnel[PrimeTunnelConfig]):
    def __init__(self, config: PrimeTunnelConfig | None = None) -> None:
        ensure_prime_auth()
        super().__init__(config)

    @contextlib.asynccontextmanager
    async def expose(self, port: int) -> AsyncIterator[str]:
        """Bridge the host `port` to a public URL via prime_tunnel (frpc). Tunnel creation
        is network-bound and rate-capped (512/min, run-wide via the shared
        `tunnel_limiter`), so transient failures are retried; a terminal one raises
        `TunnelError`. While held, frpc is restarted if requests stop reaching `port`, which
        must answer `HEALTH_PATH`. The tunnel is torn down on exit."""
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
        watch = asyncio.create_task(_keep_serving(client, self.on_gone))

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
