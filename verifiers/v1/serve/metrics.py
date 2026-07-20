"""Dependency-free Prometheus metrics for the v1 env-server pool."""

import asyncio
import contextlib
from collections.abc import Callable

_REQUEST_TIMEOUT_SECONDS = 5


def _metric_line(
    name: str, value: int | float, labels: dict[str, str] | None = None
) -> str:
    if labels:
        escaped = {
            key: value.replace("\\", "\\\\").replace('"', '\\"')
            for key, value in labels.items()
        }
        label_text = ",".join(f'{key}="{value}"' for key, value in escaped.items())
        return f"{name}{{{label_text}}} {value}"
    return f"{name} {value}"


class PoolMetrics:
    """Mutable broker-side metrics state rendered on the broker event loop."""

    def __init__(self, configured_workers: int | None) -> None:
        self.configured_workers = configured_workers
        self.request_total = 0
        self.request_latency_seconds_count = 0
        self.request_latency_seconds_sum = 0.0
        self.pending_depth = 0
        self.active_rollouts = 0
        self.workers: list[dict] = []

    def render(self) -> str:
        lines = [
            "# HELP verifiers_v1_env_active_rollouts Active rollout slots in the pool.",
            "# TYPE verifiers_v1_env_active_rollouts gauge",
            _metric_line("verifiers_v1_env_active_rollouts", self.active_rollouts),
            "# HELP verifiers_v1_env_worker_active_rollouts Active rollout slots per worker.",
            "# TYPE verifiers_v1_env_worker_active_rollouts gauge",
        ]
        lines.extend(
            _metric_line(
                "verifiers_v1_env_worker_active_rollouts",
                worker["active"],
                {"worker_index": str(worker["index"])},
            )
            for worker in self.workers
        )
        lines.extend(
            [
                "# HELP verifiers_v1_env_workers Current worker processes.",
                "# TYPE verifiers_v1_env_workers gauge",
                _metric_line("verifiers_v1_env_workers", len(self.workers)),
                "# HELP verifiers_v1_env_configured_workers Configured worker limit; zero means unbounded.",
                "# TYPE verifiers_v1_env_configured_workers gauge",
                _metric_line(
                    "verifiers_v1_env_configured_workers",
                    self.configured_workers or 0,
                ),
                "# HELP verifiers_v1_env_pending_requests Requests currently awaiting a worker reply.",
                "# TYPE verifiers_v1_env_pending_requests gauge",
                _metric_line("verifiers_v1_env_pending_requests", self.pending_depth),
                "# HELP verifiers_v1_env_requests_total Requests dispatched to workers.",
                "# TYPE verifiers_v1_env_requests_total counter",
                _metric_line("verifiers_v1_env_requests_total", self.request_total),
                "# HELP verifiers_v1_env_request_latency_seconds Request latency from dispatch to worker reply.",
                "# TYPE verifiers_v1_env_request_latency_seconds summary",
                _metric_line(
                    "verifiers_v1_env_request_latency_seconds_count",
                    self.request_latency_seconds_count,
                ),
                _metric_line(
                    "verifiers_v1_env_request_latency_seconds_sum",
                    self.request_latency_seconds_sum,
                ),
                "",
            ]
        )
        return "\n".join(lines)


class MetricsServer:
    """Minimal HTTP server exposing one Prometheus text endpoint."""

    def __init__(
        self,
        address: str,
        port: int,
        render: Callable[[], str],
    ) -> None:
        self.address = address
        self.port = port
        self.render = render
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self.address, self.port
        )
        sockets = self._server.sockets or []
        if sockets:
            host, port, *_ = sockets[0].getsockname()
            self.address = str(host)
            self.port = int(port)

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=_REQUEST_TIMEOUT_SECONDS
            )
            method, path, *_ = request_line.decode("ascii", "replace").split()
            if method != "GET":
                status = "405 Method Not Allowed"
                body = b"method not allowed\n"
            elif path.split("?", 1)[0] != "/metrics":
                status = "404 Not Found"
                body = b"not found\n"
            else:
                status = "200 OK"
                body = self.render().encode()
            headers = (
                f"HTTP/1.1 {status}\r\n"
                "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode()
            writer.write(headers + body)
            await writer.drain()
        except asyncio.TimeoutError:
            return
        except (ValueError, UnicodeError):
            writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            with contextlib.suppress(Exception):
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
