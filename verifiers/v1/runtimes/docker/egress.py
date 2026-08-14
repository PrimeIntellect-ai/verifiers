"""Authenticated host bridge and HTTP(S) policy proxy for container runtimes."""

import asyncio
import base64
import contextlib
import hmac
import secrets
import socket
import ssl
import struct
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import proxy_bypass_environment

import h11

from verifiers.v1.configs.runtime import network_rule_matches

HOST_ALIAS = "vf.host.internal"
_HEADER_TIMEOUT = 10
_IO_TIMEOUT = 300


async def _read(reader: asyncio.StreamReader) -> bytes:
    return await asyncio.wait_for(reader.read(1 << 16), _IO_TIMEOUT)


async def _drain(writer: asyncio.StreamWriter) -> None:
    await asyncio.wait_for(writer.drain(), _IO_TIMEOUT)


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
                    rule.lower().startswith(f"{scheme}://")
                    and network_rule_matches(rule, scheme, host, port)
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
        if any(
            network_rule_matches(route, scheme, host, port) for route in self.routes
        ):
            return True
        # The proxy dials from the host, so only framework routes may use host loopback.
        with contextlib.suppress(ValueError):
            if ip_address(hostname).is_loopback:
                return False
        if any(network_rule_matches(rule, scheme, host, port) for rule in self.block):
            return False
        return any(
            network_rule_matches(rule, scheme, host, port) for rule in self.allow
        )


@dataclass(frozen=True)
class _UpstreamProxy:
    host: str
    port: int
    scheme: str
    authorization: bytes | None
    username: str | None
    password: str | None

    @property
    def remote_dns(self) -> bool:
        return self.scheme in ("http", "https", "socks4a", "socks5h")

    @classmethod
    def parse(cls, url: str) -> "_UpstreamProxy":
        parsed = urlsplit(url if "://" in url else f"http://{url}")
        schemes = {"http", "https", "socks4", "socks4a", "socks5", "socks5h"}
        if parsed.scheme not in schemes or parsed.hostname is None:
            raise ValueError(f"unsupported upstream proxy URL: {url!r}")
        username = unquote(parsed.username) if parsed.username is not None else None
        password = unquote(parsed.password or "") if username is not None else None
        authorization = None
        if username is not None and parsed.scheme in ("http", "https"):
            credentials = f"{username}:{password}"
            authorization = b"Basic " + base64.b64encode(credentials.encode())
        return cls(
            parsed.hostname,
            parsed.port or {"http": 80, "https": 443}.get(parsed.scheme, 1080),
            parsed.scheme,
            authorization,
            username,
            password,
        )


@dataclass(frozen=True)
class _ProxyRoute:
    upstream: _UpstreamProxy | None = None
    no_proxy: str | None = None


def _proxy_bypass(host: str, no_proxy: str) -> bool:
    if proxy_bypass_environment(host, {"no": no_proxy}):
        return True
    hostname = urlsplit(f"//{host}").hostname
    if hostname is None:
        return False
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    for entry in no_proxy.split(","):
        with contextlib.suppress(ValueError):
            if address in ip_network(entry.strip(), strict=False):
                return True
    return False


async def _connect_socks(
    proxy: _UpstreamProxy,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    host: str,
    port: int,
) -> None:
    username = (proxy.username or "").encode()
    if proxy.scheme in ("socks5", "socks5h"):
        methods = b"\x00\x02" if proxy.username is not None else b"\x00"
        writer.write(bytes((5, len(methods))) + methods)
        await _drain(writer)
        version, method = await asyncio.wait_for(reader.readexactly(2), _HEADER_TIMEOUT)
        if version != 5 or method == 0xFF:
            raise ConnectionError("SOCKS5 proxy rejected authentication methods")
        if method == 2:
            password = (proxy.password or "").encode()
            if len(username) > 255 or len(password) > 255:
                raise ValueError("SOCKS5 proxy credentials are too long")
            writer.write(
                bytes((1, len(username)))
                + username
                + bytes((len(password),))
                + password
            )
            await _drain(writer)
            if (
                await asyncio.wait_for(reader.readexactly(2), _HEADER_TIMEOUT)
                != b"\x01\x00"
            ):
                raise ConnectionError("SOCKS5 proxy rejected credentials")
        elif method != 0:
            raise ConnectionError(
                f"SOCKS5 proxy selected unsupported method ({method})"
            )
        try:
            address = ip_address(host)
            encoded_host = bytes((1 if address.version == 4 else 4,)) + address.packed
        except ValueError:
            encoded = host.encode("idna")
            if len(encoded) > 255:
                raise ValueError("SOCKS5 target hostname is too long")
            encoded_host = bytes((3, len(encoded))) + encoded
        writer.write(b"\x05\x01\x00" + encoded_host + struct.pack("!H", port))
        await _drain(writer)
        version, status, _, address_type = await asyncio.wait_for(
            reader.readexactly(4), _HEADER_TIMEOUT
        )
        if version != 5 or status != 0:
            raise ConnectionError(f"SOCKS5 proxy rejected connection ({status})")
        length = {1: 4, 4: 16}.get(address_type)
        if address_type == 3:
            length = (await asyncio.wait_for(reader.readexactly(1), _HEADER_TIMEOUT))[0]
        if length is None:
            raise ConnectionError("SOCKS5 proxy returned an invalid address")
        await asyncio.wait_for(reader.readexactly(length + 2), _HEADER_TIMEOUT)
        return

    suffix = b""
    try:
        address = ip_address(host)
        if address.version != 4:
            raise ValueError
        encoded_host = address.packed
    except ValueError:
        if proxy.scheme != "socks4a":
            raise ConnectionError("SOCKS4 requires an IPv4 target")
        encoded_host = b"\x00\x00\x00\x01"
        suffix = host.encode("idna") + b"\x00"
    writer.write(
        b"\x04\x01"
        + struct.pack("!H", port)
        + encoded_host
        + username
        + b"\x00"
        + suffix
    )
    await _drain(writer)
    response = await asyncio.wait_for(reader.readexactly(8), _HEADER_TIMEOUT)
    if response[1] != 90:
        raise ConnectionError(f"SOCKS4 proxy rejected connection ({response[1]})")


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
        header = await asyncio.wait_for(reader.readexactly(5), _HEADER_TIMEOUT)
        length = int.from_bytes(header[3:5])
        if header[0] != 22 or length > (1 << 14) + 2048:
            raise ValueError("expected a TLS handshake record")
        payload = await asyncio.wait_for(reader.readexactly(length), _HEADER_TIMEOUT)
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
        self.token = secrets.token_urlsafe(32)
        self._authorization = b"Basic " + base64.b64encode(
            f"verifiers:{self.token}".encode()
        )
        self._routes = {self._authorization: _ProxyRoute()}
        self._route_tokens: dict[tuple[str, str], str] = {}
        self.server: asyncio.Server | None = None
        self.port = 0

    def token_for(self, upstream: str | None, no_proxy: str | None = None) -> str:
        if not upstream:
            return self.token
        key = (upstream, no_proxy or "")
        if token := self._route_tokens.get(key):
            return token
        token = secrets.token_urlsafe(32)
        authorization = b"Basic " + base64.b64encode(f"verifiers:{token}".encode())
        try:
            parsed = _UpstreamProxy.parse(upstream)
        except ValueError:
            parsed = None
        self._routes[authorization] = _ProxyRoute(parsed, no_proxy)
        self._route_tokens[key] = token
        return token

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
            head = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), _HEADER_TIMEOUT
            )
            client = h11.Connection(h11.SERVER)
            client.receive_data(head)
            request = client.next_event()
            if not isinstance(request, h11.Request):
                raise TypeError("expected an HTTP request")
            authorization = next(
                (
                    value
                    for name, value in request.headers
                    if name.lower() == b"proxy-authorization"
                ),
                b"",
            )
            selected = object()
            route: _ProxyRoute | object = selected
            for expected, candidate in self._routes.items():
                if hmac.compare_digest(authorization, expected):
                    route = candidate
                    break
            if route is selected:
                response_started = True
                writer.write(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                    b'Proxy-Authenticate: Basic realm="verifiers"\r\n'
                    b"Content-Length: 0\r\n\r\n"
                )
                await _drain(writer)
                return
            assert isinstance(route, _ProxyRoute)
            upstream_proxy = route.upstream
            method = request.method.decode("ascii")
            target = request.target.decode("ascii")
            connect = method == "CONNECT"
            if connect:
                parsed = urlsplit(f"//{target}")
                host, port = parsed.hostname or "", parsed.port or 443
                # Some HTTP clients tunnel plain HTTP through CONNECT. Only an
                # explicit framework route can identify that otherwise-ambiguous
                # tunnel without broadening user-configured egress.
                scheme = (
                    "http"
                    if any(
                        route.lower().startswith("http://")
                        and network_rule_matches(route, "http", host, port)
                        for route in self.policy.routes
                    )
                    else "https"
                )
            else:
                parsed = urlsplit(target)
                scheme = parsed.scheme.lower()
                host = parsed.hostname or ""
                port = parsed.port or (443 if scheme == "https" else 80)
            permitted = (connect or scheme == "http") and self.policy.permits(
                scheme, host, port, connect=connect
            )
            framework = any(
                network_rule_matches(route, scheme, host, port)
                for route in self.policy.routes
            )
            authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            if (framework and host.lower() == HOST_ALIAS) or (
                upstream_proxy is not None
                and route.no_proxy is not None
                and _proxy_bypass(authority, route.no_proxy)
            ):
                upstream_proxy = None
            target_addresses = []
            proxy_target = None
            if permitted:
                dial_host = "127.0.0.1" if host.lower() == HOST_ALIAS else host
                if (
                    upstream_proxy is None
                    or not self.policy.allow_non_global
                    or not upstream_proxy.remote_dns
                ):
                    target_addresses = await asyncio.wait_for(
                        asyncio.get_running_loop().getaddrinfo(
                            dial_host, port, type=socket.SOCK_STREAM
                        ),
                        _IO_TIMEOUT,
                    )
                if not framework and not self.policy.allow_non_global:
                    for *_, address in target_addresses:
                        resolved = ip_address(address[0])
                        mapped = getattr(resolved, "ipv4_mapped", None)
                        if not (mapped or resolved).is_global:
                            permitted = False
                            break
                if permitted and upstream_proxy is not None and target_addresses:
                    # Keep an upstream proxy from resolving an allowed name to a private address.
                    candidates = [address[4][0] for address in target_addresses]
                    if upstream_proxy.scheme in ("socks4", "socks4a"):
                        candidates = [
                            address
                            for address in candidates
                            if ip_address(address).version == 4
                        ]
                    if (
                        not upstream_proxy.remote_dns
                        or not self.policy.allow_non_global
                    ):
                        if not candidates:
                            permitted = False
                        else:
                            proxy_target = candidates[0]
            if not permitted:
                response_started = True
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await _drain(writer)
                return
            assert upstream_proxy is None or isinstance(upstream_proxy, _UpstreamProxy)
            connect_host = (
                upstream_proxy.host if upstream_proxy is not None else dial_host
            )
            connect_port = upstream_proxy.port if upstream_proxy is not None else port
            addresses = (
                await asyncio.wait_for(
                    asyncio.get_running_loop().getaddrinfo(
                        connect_host, connect_port, type=socket.SOCK_STREAM
                    ),
                    _IO_TIMEOUT,
                )
                if upstream_proxy is not None
                else target_addresses
            )
            for family, _, _, _, address in addresses:
                try:
                    tls = (
                        ssl.create_default_context()
                        if upstream_proxy and upstream_proxy.scheme == "https"
                        else None
                    )
                    upstream_reader, upstream_writer = await asyncio.wait_for(
                        asyncio.open_connection(
                            address[0],
                            address[1],
                            family=family,
                            flags=socket.AI_NUMERICHOST,
                            ssl=tls,
                            server_hostname=(
                                upstream_proxy.host
                                if upstream_proxy is not None and tls
                                else None
                            ),
                        ),
                        _HEADER_TIMEOUT,
                    )
                    break
                except (OSError, TimeoutError):
                    continue
            if upstream_reader is None or upstream_writer is None:
                raise ConnectionError(f"could not connect to {host}:{port}")
            socks = upstream_proxy is not None and upstream_proxy.scheme.startswith(
                "socks"
            )
            if socks:
                await _connect_socks(
                    upstream_proxy,
                    upstream_reader,
                    upstream_writer,
                    proxy_target or host,
                    port,
                )
            if connect:
                if upstream_proxy is not None and not socks:
                    if proxy_target is None:
                        authority = request.target
                    else:
                        target_host = (
                            f"[{proxy_target}]" if ":" in proxy_target else proxy_target
                        )
                        authority = f"{target_host}:{port}".encode("ascii")
                    headers = [
                        b"CONNECT " + authority + b" HTTP/1.1",
                        b"Host: " + authority,
                    ]
                    if upstream_proxy.authorization is not None:
                        headers.append(
                            b"Proxy-Authorization: " + upstream_proxy.authorization
                        )
                    upstream_writer.write(b"\r\n".join(headers) + b"\r\n\r\n")
                    await _drain(upstream_writer)
                    response_head = await asyncio.wait_for(
                        upstream_reader.readuntil(b"\r\n\r\n"), _HEADER_TIMEOUT
                    )
                    response = h11.Connection(h11.CLIENT)
                    response.receive_data(response_head)
                    status = response.next_event()
                    if (
                        not isinstance(status, h11.Response)
                        or not 200 <= status.status_code < 300
                    ):
                        raise ConnectionError("upstream proxy rejected CONNECT")
                response_started = True
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await _drain(writer)
                if scheme == "https":
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
                    await _drain(upstream_writer)
                await _relay(reader, writer, upstream_reader, upstream_writer)
            else:
                if upstream_proxy is None or socks:
                    path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                elif proxy_target is None:
                    path = request.target
                else:
                    target_host = (
                        f"[{proxy_target}]" if ":" in proxy_target else proxy_target
                    )
                    if port != (443 if scheme == "https" else 80):
                        target_host = f"{target_host}:{port}"
                    path = urlunsplit(
                        (scheme, target_host, parsed.path or "/", parsed.query, "")
                    )
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
                if (
                    upstream_proxy is not None
                    and not socks
                    and upstream_proxy.authorization is not None
                ):
                    headers.append(
                        (b"Proxy-Authorization", upstream_proxy.authorization)
                    )
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
                await _drain(upstream_writer)
                if any(
                    name.lower() == b"expect" and value.lower() == b"100-continue"
                    for name, value in request.headers
                ):
                    writer.write(
                        client.send(
                            h11.InformationalResponse(status_code=100, headers=[])
                        )
                    )
                    await _drain(writer)
                while True:
                    event = client.next_event()
                    if event is h11.NEED_DATA:
                        client.receive_data(await _read(reader))
                    elif isinstance(event, h11.Data):
                        upstream_writer.write(upstream.send(event))
                        await _drain(upstream_writer)
                    elif isinstance(event, h11.EndOfMessage):
                        upstream_writer.write(upstream.send(event))
                        break
                    else:
                        raise ValueError("incomplete HTTP request body")
                await _drain(upstream_writer)
                # Plain HTTP gets exactly one policy check and one request. Never copy
                # pipelined bytes into the first request's already-selected upstream.
                while chunk := await _read(upstream_reader):
                    response_started = True
                    writer.write(chunk)
                    await _drain(writer)
        except Exception:  # noqa: BLE001 - proxy failures become a generic 502
            if not response_started:
                with contextlib.suppress(Exception):
                    writer.write(
                        b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"
                    )
                    await _drain(writer)
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
            while chunk := await _read(reader):
                writer.write(chunk)
                await _drain(writer)
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
