"""Egress filtering for network-restricted runtimes: a minimal CONNECT proxy.

The L3 cut (internal network on Linux, route pinning on macOS) leaves an isolated
container no direct route off its network — except the address host services bind. An
`EgressProxy` bound there is therefore the *only* way out, and it enforces the
runtime's allow/block host patterns per CONNECT. One shared proxy per ruleset per
process (created on first use, kept around, like the offline network). Clients opt in
via the standard `HTTP(S)_PROXY` env vars the runtime injects; `NO_PROXY` covers the
interception and host MCP addresses so framework traffic never hairpins through it.
Filtering thus applies to proxy-aware clients; the L3 cut is the hard guarantee that
nothing else leaves directly.
"""

import asyncio
import contextlib
import fnmatch
import logging
from collections.abc import Callable
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


def host_matcher(
    allow: list[str], block: list[str], default_allow: bool
) -> Callable[[str], bool]:
    """A predicate over target hosts: block patterns win; then either default-allow
    (broad access minus the block list) or default-deny (only the allow list).
    Patterns are fnmatch wildcards on the lowercased host; "*.example.com" also
    matches the apex "example.com"."""
    allows = [p.lower() for p in allow]
    blocks = [p.lower() for p in block]

    def matches(host: str) -> bool:
        h = host.lower()
        if any(_match(h, p) for p in blocks):
            return False
        return default_allow or any(_match(h, p) for p in allows)

    return matches


def _match(host: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(host, pattern) or (
        pattern.startswith("*.") and host == pattern[2:]
    )


class EgressProxy:
    """A minimal forward proxy enforcing a host matcher: CONNECT for HTTPS,
    absolute-form forwarding for plain HTTP (both filtered the same way).
    `dial_map` rewrites target hosts before dialing — names a container uses for
    the host (host.docker.internal) don't resolve *on* the host (loopback there)."""

    def __init__(
        self,
        allow: list[str],
        block: list[str],
        default_allow: bool,
        dial_map: dict[str, str] | None = None,
    ) -> None:
        self.match = host_matcher(allow, block, default_allow)
        self.dial_map = {k.lower(): v for k, v in (dial_map or {}).items()}
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(self, bind_host: str) -> int:
        self.server = await asyncio.start_server(self._handle, bind_host, 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            method, target, _ = (
                head.split(b"\r\n", 1)[0].decode("latin-1").split(" ", 2)
            )
            if method == "CONNECT":
                host, sep, port = target.rpartition(":")
                if not sep:
                    host, port = target, "443"
                port = int(port)
            else:  # plain HTTP in absolute-form (origins accept it as-is)
                parsed = urlsplit(target)
                host, port = parsed.hostname or "", parsed.port or 80
            if not self.match(host):
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            upstream_reader, upstream_writer = await asyncio.open_connection(
                self.dial_map.get(host.lower(), host), port
            )
            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                upstream_writer.write(head)
                await upstream_writer.drain()
            await _relay(reader, writer, upstream_reader, upstream_writer)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
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
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    left = asyncio.create_task(pipe(client_reader, upstream_writer))
    right = asyncio.create_task(pipe(upstream_reader, client_writer))
    await asyncio.wait({left, right}, return_when=asyncio.FIRST_COMPLETED)
    left.cancel()
    right.cancel()
    await asyncio.gather(left, right, return_exceptions=True)


_proxies: dict[tuple, EgressProxy] = {}


async def ensure_egress_proxy(
    allow: list[str],
    block: list[str],
    *,
    default_allow: bool,
    bind_host: str,
    dial_map: dict[str, str] | None = None,
) -> int:
    """The shared proxy for one ruleset (created on first use, kept around), bound
    where isolated containers can reach it; returns its port."""
    key = (tuple(allow), tuple(block), default_allow, bind_host)
    proxy = _proxies.get(key)
    if proxy is None:
        proxy = EgressProxy(allow, block, default_allow, dial_map)
        await proxy.start(bind_host)
        _proxies[key] = proxy
        logger.info(
            "egress proxy up: bind=%s:%d default_allow=%s allow=%s block=%s",
            bind_host,
            proxy.port,
            default_allow,
            allow,
            block,
        )
    return proxy.port
