"""In-process HTTP(S) policy proxy for network-filtered Docker runtimes."""

import asyncio
import contextlib
import fnmatch
import socket
import struct
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
        hostname = host.lower().rstrip(".")
        # `localhost` is container-local and bypasses this host-side proxy. Never dial
        # the host's namesake address, even if a colocated service appears in routes.
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return False
        # Framework routes are invariants, not user egress, so they cannot be blocked.
        if any(_rule_matches(route, scheme, host, port) for route in self.routes):
            return True
        # The proxy dials from the host, so only framework routes may use host loopback.
        with contextlib.suppress(ValueError):
            if ip_address(hostname).is_loopback:
                return False
        if any(_rule_matches(rule, scheme, host, port) for rule in self.block):
            return False
        return self.default_allow or any(
            _rule_matches(rule, scheme, host, port) for rule in self.allow
        )


def _server_name(client_hello: bytes) -> str | None:
    """Read the SNI hostname from a complete TLS ClientHello handshake body."""
    data = memoryview(client_hello)
    offset = 34  # legacy_version + random
    offset += 1 + data[offset]  # session id
    cipher_length = struct.unpack_from("!H", data, offset)[0]
    offset += 2 + cipher_length
    offset += 1 + data[offset]  # compression methods
    if offset == len(data):
        return None
    extensions_length = struct.unpack_from("!H", data, offset)[0]
    offset += 2
    end = offset + extensions_length
    if end != len(data):
        raise ValueError("malformed TLS extensions")
    while offset < end:
        extension_type, extension_length = struct.unpack_from("!HH", data, offset)
        offset += 4
        extension = data[offset : offset + extension_length]
        offset += extension_length
        if len(extension) != extension_length:
            raise ValueError("malformed TLS extension")
        if extension_type != 0:
            continue
        names_length = struct.unpack_from("!H", extension, 0)[0]
        if names_length != len(extension) - 2:
            raise ValueError("malformed TLS server names")
        name_offset = 2
        while name_offset < len(extension):
            name_type = extension[name_offset]
            name_length = struct.unpack_from("!H", extension, name_offset + 1)[0]
            name_offset += 3
            name = bytes(extension[name_offset : name_offset + name_length])
            name_offset += name_length
            if name_offset > len(extension):
                raise ValueError("malformed TLS server name")
            if name_type == 0:
                hostname = name.decode("ascii").lower().rstrip(".")
                if not hostname or "\0" in hostname:
                    raise ValueError("invalid TLS server name")
                return hostname
        return None
    return None


async def _read_client_hello(
    reader: asyncio.StreamReader,
) -> tuple[bytes, str | None]:
    records = bytearray()
    handshake = bytearray()
    expected: int | None = None
    while expected is None or len(handshake) < expected:
        header = await asyncio.wait_for(reader.readexactly(5), 10)
        length = int.from_bytes(header[3:5])
        if header[0] != 22 or length > (1 << 14) + 2048:
            raise ValueError("expected a TLS handshake record")
        payload = await asyncio.wait_for(reader.readexactly(length), 10)
        records.extend(header)
        records.extend(payload)
        handshake.extend(payload)
        if len(records) > 1 << 20:
            raise ValueError("TLS ClientHello is too large")
        if expected is None and len(handshake) >= 4:
            if handshake[0] != 1:
                raise ValueError("expected a TLS ClientHello")
            expected = 4 + int.from_bytes(handshake[1:4])
            if expected > 1 << 20:
                raise ValueError("TLS ClientHello is too large")
    return bytes(records), _server_name(bytes(handshake[4:expected]))


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
        upstream_reader: asyncio.StreamReader | None = None
        upstream_writer: asyncio.StreamWriter | None = None
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
            permitted = scheme in ("http", "https") and self.policy.permits(
                scheme, host, port, connect=method == "CONNECT"
            )
            addresses = []
            if permitted:
                addresses = await asyncio.get_running_loop().getaddrinfo(
                    host, port, type=socket.SOCK_STREAM
                )
                framework = any(
                    _rule_matches(route, scheme, host, port)
                    for route in self.policy.routes
                )
                for *_, address in addresses:
                    resolved = ip_address(address[0])
                    mapped = getattr(resolved, "ipv4_mapped", None)
                    host_only = (
                        resolved.is_loopback
                        or resolved.is_link_local
                        or (mapped and (mapped.is_loopback or mapped.is_link_local))
                    )
                    if (
                        resolved.is_unspecified
                        or (mapped and mapped.is_unspecified)
                        or (not framework and host_only)
                    ):
                        permitted = False
                        break
            if not permitted:
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            for family, _, _, _, address in addresses:
                try:
                    upstream_reader, upstream_writer = await asyncio.open_connection(
                        address[0],
                        address[1],
                        family=family,
                        flags=socket.AI_NUMERICHOST,
                    )
                    break
                except OSError:
                    continue
            if upstream_reader is None or upstream_writer is None:
                raise ConnectionError(f"could not connect to {host}:{port}")
            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                client_hello, server_name = await _read_client_hello(reader)
                if server_name is None:
                    with contextlib.suppress(ValueError):
                        ip_address(host)
                        server_name = host
                if server_name is None or not self.policy.permits(
                    "https", server_name, port, connect=True
                ):
                    return
                upstream_writer.write(client_hello)
                await upstream_writer.drain()
            else:
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                authority = f"[{host}]" if ":" in host else host
                if port != (443 if scheme == "https" else 80):
                    authority = f"{authority}:{port}"
                header_lines = [
                    line
                    for line in headers.split(b"\r\n")
                    if line.partition(b":")[0].strip().lower() != b"host"
                ]
                headers = f"Host: {authority}\r\n".encode("latin-1") + b"\r\n".join(
                    header_lines
                )
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
            if upstream_writer is not None:
                upstream_writer.close()
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
