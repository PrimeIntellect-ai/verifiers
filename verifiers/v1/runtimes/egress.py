"""In-process HTTP(S) policy proxy for network-filtered Docker runtimes."""

import asyncio
import contextlib
import fnmatch
import socket
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit


def _rule_matches(rule: str, scheme: str, host: str, port: int) -> bool:
    """Match a host pattern or URL origin. URL paths are intentionally ignored."""
    value = rule.lower().rstrip("/")
    parsed = urlsplit(value if "://" in value else f"//{value}")
    pattern = (parsed.hostname or "").rstrip(".")
    if not pattern or (parsed.scheme and parsed.scheme != scheme):
        return False
    rule_port = parsed.port
    if parsed.scheme and rule_port is None:
        rule_port = 443 if parsed.scheme == "https" else 80
    if rule_port is not None and rule_port != port:
        return False
    host = host.lower().rstrip(".")
    return fnmatch.fnmatchcase(host, pattern) or (
        pattern.startswith("*.") and host == pattern[2:]
    )


@dataclass
class NetworkPolicy:
    allow: list[str]
    block: list[str]
    routes: list[str]
    default_allow: bool

    def permits(
        self, scheme: str, host: str, port: int, *, connect: bool = False
    ) -> bool:
        if (
            connect
            and port != 443
            and not any(
                urlsplit(rule.lower()).scheme == "https"
                and _rule_matches(rule, scheme, host, port)
                for rule in [*self.routes, *self.allow]
            )
        ):
            return False
        # Framework routes are invariants, not user egress, so they cannot be blocked.
        if any(_rule_matches(route, scheme, host, port) for route in self.routes):
            return True
        # The proxy dials from the host, so only framework routes may use host loopback.
        hostname = host.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return False
        with contextlib.suppress(ValueError):
            if ip_address(hostname).is_loopback:
                return False
        if any(_rule_matches(rule, scheme, host, port) for rule in self.block):
            return False
        return self.default_allow or any(
            _rule_matches(rule, scheme, host, port) for rule in self.allow
        )


class EgressProxy:
    def __init__(self, policy: NetworkPolicy) -> None:
        self.policy = policy
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(
        self, bind_host: str | None = None, *, listener: socket.socket | None = None
    ) -> None:
        if listener is None:
            self.server = await asyncio.start_server(self._handle, bind_host, 0)
        else:
            self.server = await asyncio.start_server(self._handle, sock=listener)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        self.server = None

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            request, headers = head.split(b"\r\n", 1)
            method, target, version = request.decode("latin-1").split(" ", 2)
            if method == "CONNECT":
                parsed = urlsplit(f"//{target}")
                scheme = "https"
                host, port = parsed.hostname or "", parsed.port or 443
            else:
                parsed = urlsplit(target)
                scheme = parsed.scheme.lower()
                host = parsed.hostname or ""
                port = parsed.port or (443 if scheme == "https" else 80)
            if scheme not in ("http", "https") or not self.policy.permits(
                scheme, host, port, connect=method == "CONNECT"
            ):
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                upstream_writer.write(
                    f"{method} {path} {version}\r\n".encode("latin-1") + headers
                )
                await upstream_writer.drain()
            await _relay(reader, writer, upstream_reader, upstream_writer)
        except Exception:
            with contextlib.suppress(Exception):
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
        finally:
            writer.close()


async def _relay(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while chunk := await reader.read(1 << 16):
                writer.write(chunk)
                await writer.drain()
        finally:
            writer.close()

    tasks = {
        asyncio.create_task(pipe(client_reader, upstream_writer)),
        asyncio.create_task(pipe(upstream_reader, client_writer)),
    }
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
