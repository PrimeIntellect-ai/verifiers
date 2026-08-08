"""Prometheus text-format metrics for env server stats."""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.serve.server.env_router import EnvRouter, EnvRouterStats

from verifiers.utils.async_utils import EventLoopLagStats

logger = logging.getLogger(__name__)


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _lag_quantiles(lag: EventLoopLagStats) -> dict[str, float]:
    return {
        "p50": lag.median,
        "p95": lag.p95,
        "p99": lag.p99,
    }


def render_prometheus_text(
    router_stats: "EnvRouterStats", *, env_id: str, version: str
) -> str:
    """Render an EnvRouterStats snapshot as Prometheus text exposition format."""
    escaped_env_id = _escape_label_value(env_id)
    escaped_version = _escape_label_value(version)
    lines: list[str] = [
        "# HELP verifiers_env_server_info Env server build and identity labels.",
        "# TYPE verifiers_env_server_info gauge",
        (
            "verifiers_env_server_info"
            f'{{env_id="{escaped_env_id}",version="{escaped_version}"}} 1'
        ),
        "# HELP verifiers_env_active_tasks Total active rollouts across workers.",
        "# TYPE verifiers_env_active_tasks gauge",
        f"verifiers_env_active_tasks {router_stats.active_tasks}",
        "# HELP verifiers_env_workers_total Configured worker count.",
        "# TYPE verifiers_env_workers_total gauge",
        f"verifiers_env_workers_total {router_stats.num_workers}",
        "# HELP verifiers_env_worker_active_tasks Active rollouts per worker.",
        "# TYPE verifiers_env_worker_active_tasks gauge",
    ]

    for worker_id, worker_stats in sorted(router_stats.workers.items()):
        active_tasks = worker_stats.active_tasks if worker_stats is not None else 0
        lines.append(
            f'verifiers_env_worker_active_tasks{{worker_id="{worker_id}"}} {active_tasks}'
        )

    lines.extend(
        [
            "# HELP verifiers_env_loop_lag_seconds Asyncio event loop lag in seconds.",
            "# TYPE verifiers_env_loop_lag_seconds gauge",
        ]
    )
    for quantile, value in _lag_quantiles(router_stats.lag).items():
        lines.append(
            "verifiers_env_loop_lag_seconds"
            f'{{worker_id="router",quantile="{quantile}"}} {value}'
        )
    for worker_id, worker_stats in sorted(router_stats.workers.items()):
        if worker_stats is None:
            continue
        for quantile, value in _lag_quantiles(worker_stats.lag).items():
            lines.append(
                "verifiers_env_loop_lag_seconds"
                f'{{worker_id="{worker_id}",quantile="{quantile}"}} {value}'
            )

    return "\n".join(lines) + "\n"


class MetricsServer:
    """Asyncio HTTP server serving /metrics in Prometheus text format."""

    def __init__(
        self, router: "EnvRouter", *, env_id: str, version: str, port: int
    ) -> None:
        self.router = router
        self.env_id = env_id
        self.version = version
        self.port = port
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(self.handle, "0.0.0.0", self.port)
        logger.info(f"Metrics server listening on http://0.0.0.0:{self.port}/metrics")

    async def close(self) -> None:
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        self.server = None

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            parts = request_line.split()
            if len(parts) < 2 or parts[0] != b"GET" or parts[1] != b"/metrics":
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            else:
                body = render_prometheus_text(
                    self.router.stats,
                    env_id=self.env_id,
                    version=self.version,
                ).encode("utf-8")
                headers = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                )
                writer.write(headers + body)
            await writer.drain()
        except Exception:
            logger.exception("Metrics server request handling failed")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
