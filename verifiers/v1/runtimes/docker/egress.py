"""In-process HTTP(S) policy proxy for network-filtered Docker runtimes."""

import asyncio
import contextlib
import fnmatch
import socket
import ssl
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit

import h11

HOST_ALIAS = "vf.host.internal"


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
    allow_non_global: bool = False  # trusted setup only

    def permits(
        self, scheme: str, host: str, port: int, *, connect: bool = False
    ) -> bool:
        if (
            connect
            and port != 443
            and not any(
                rule == "*"
                or (
                    urlsplit(rule.lower()).scheme == "https"
                    and _rule_matches(rule, scheme, host, port)
                )
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
        return any(_rule_matches(rule, scheme, host, port) for rule in self.allow)


async def _read_client_hello(
    reader: asyncio.StreamReader,
) -> tuple[bytes, str | None]:
    """Buffer TLS records through OpenSSL until it exposes the ClientHello SNI."""
    server_name: str | None = None

    def capture_sni(_: ssl.SSLObject, name: str | None, __: ssl.SSLContext) -> None:
        nonlocal server_name
        server_name = name

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.set_servername_callback(capture_sni)
    incoming = ssl.MemoryBIO()
    tls = context.wrap_bio(incoming, ssl.MemoryBIO(), server_side=True)
    records = bytearray()
    while server_name is None:
        header = await asyncio.wait_for(reader.readexactly(5), 10)
        length = int.from_bytes(header[3:5])
        if header[0] != 22 or length > (1 << 14) + 2048:
            raise ValueError("expected a TLS handshake record")
        payload = await asyncio.wait_for(reader.readexactly(length), 10)
        records.extend(header)
        records.extend(payload)
        if len(records) > 1 << 20:
            raise ValueError("TLS ClientHello is too large")
        incoming.write(header + payload)
        try:
            tls.do_handshake()
        except ssl.SSLWantReadError:
            continue
        except ssl.SSLError:
            break
        break
    if server_name is not None:
        server_name = server_name.lower().rstrip(".")
    return bytes(records), server_name


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
        response_started = False
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            client = h11.Connection(h11.SERVER)
            client.receive_data(head)
            request = client.next_event()
            if not isinstance(request, h11.Request):
                raise ValueError("expected an HTTP request")
            method = request.method.decode("ascii")
            target = request.target.decode("ascii")
            connect = method == "CONNECT"
            if connect:
                parsed = urlsplit(f"//{target}")
                scheme = "https"
                host, port = parsed.hostname or "", parsed.port or 443
            else:
                parsed = urlsplit(target)
                scheme = parsed.scheme.lower()
                host = parsed.hostname or ""
                port = parsed.port or (443 if scheme == "https" else 80)
            permitted = scheme in ("http", "https") and self.policy.permits(
                scheme, host, port, connect=connect
            )
            addresses = []
            if permitted:
                dial_host = "127.0.0.1" if host.lower() == HOST_ALIAS else host
                addresses = await asyncio.get_running_loop().getaddrinfo(
                    dial_host, port, type=socket.SOCK_STREAM
                )
                framework = any(
                    _rule_matches(route, scheme, host, port)
                    for route in self.policy.routes
                )
                if not framework and not self.policy.allow_non_global:
                    for *_, address in addresses:
                        resolved = ip_address(address[0])
                        mapped = getattr(resolved, "ipv4_mapped", None)
                        if not (mapped or resolved).is_global:
                            permitted = False
                            break
            if not permitted:
                response_started = True
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
            if connect:
                response_started = True
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
                await _relay(reader, writer, upstream_reader, upstream_writer)
            else:
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                authority = f"[{host}]" if ":" in host else host
                if port != (443 if scheme == "https" else 80):
                    authority = f"{authority}:{port}"
                connection_fields = {
                    field.strip().lower()
                    for name, value in request.headers
                    if name.lower() == b"connection"
                    for field in value.split(b",")
                }
                excluded = {
                    b"connection",
                    b"expect",
                    b"host",
                    b"keep-alive",
                    b"proxy-authenticate",
                    b"proxy-authorization",
                    b"proxy-connection",
                    b"te",
                    b"trailer",
                    b"upgrade",
                    *connection_fields,
                }
                headers = [
                    (name, value)
                    for name, value in request.headers
                    if name.lower() not in excluded
                ]
                upstream = h11.Connection(h11.CLIENT)
                upstream_writer.write(
                    upstream.send(
                        h11.Request(
                            method=request.method,
                            target=path,
                            headers=[
                                (b"Host", authority.encode("ascii")),
                                (b"Connection", b"close"),
                                *headers,
                            ],
                            http_version=request.http_version,
                        )
                    )
                )
                await upstream_writer.drain()
                if any(
                    name.lower() == b"expect" and value.lower() == b"100-continue"
                    for name, value in request.headers
                ):
                    writer.write(
                        client.send(
                            h11.InformationalResponse(status_code=100, headers=[])
                        )
                    )
                    await writer.drain()
                while True:
                    event = client.next_event()
                    if event is h11.NEED_DATA:
                        client.receive_data(await reader.read(1 << 16))
                    elif isinstance(event, h11.Data):
                        upstream_writer.write(upstream.send(event))
                        await upstream_writer.drain()
                    elif isinstance(event, h11.EndOfMessage):
                        upstream_writer.write(upstream.send(event))
                        break
                    else:
                        raise ValueError("incomplete HTTP request body")
                await upstream_writer.drain()
                # Plain HTTP gets exactly one policy check and one request. Never copy
                # pipelined bytes into the first request's already-selected upstream.
                while chunk := await upstream_reader.read(1 << 16):
                    response_started = True
                    writer.write(chunk)
                    await writer.drain()
        except Exception:
            if not response_started:
                with contextlib.suppress(Exception):
                    writer.write(
                        b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"
                    )
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
