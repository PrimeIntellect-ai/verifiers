"""In-process HTTP(S) proxy for Docker policy and host callbacks."""

import asyncio
import base64
import contextlib
import hmac
import secrets
import socket
import ssl
from dataclasses import dataclass, replace
from ipaddress import ip_address
from urllib.parse import urljoin, urlsplit, urlunsplit

import h11

from verifiers.v1.configs.runtime import NetworkPolicyConfig, network_rule_matches

HOST_ALIAS = "vf.host.internal"
_CALLBACK_PREFIX = "/.vf-host/"
_HEADER_TIMEOUT = 10
_IO_TIMEOUT = 300


def is_loopback_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return (getattr(address, "ipv4_mapped", None) or address).is_loopback


async def _read(
    reader: asyncio.StreamReader, timeout: float | None = _IO_TIMEOUT
) -> bytes:
    return await asyncio.wait_for(reader.read(1 << 16), timeout)


async def _drain(writer: asyncio.StreamWriter) -> None:
    await asyncio.wait_for(writer.drain(), _IO_TIMEOUT)


@dataclass
class NetworkPolicy:
    config: NetworkPolicyConfig
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
                for rule in [*self.routes, *self.config.allow]
            )
        ):
            return False
        # Loopback framework services are container-local and bypass this proxy. Host
        # callbacks use HOST_ALIAS, so proxying a loopback name would expose the host.
        if is_loopback_host(host):
            return False
        # Framework routes are invariants, not user egress, so they cannot be blocked.
        if any(
            network_rule_matches(route, scheme, host, port) for route in self.routes
        ):
            return True
        return self.config.permits(scheme, host, port)


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


@dataclass(frozen=True)
class _Callback:
    scheme: str
    host: str
    port: int
    authority: str
    host_alias: str
    forward_authorization: bool


class EgressProxy:
    def __init__(self, policy: NetworkPolicy) -> None:
        self.policy = policy
        self.token = secrets.token_urlsafe(32)
        self._authorization = b"Basic " + base64.b64encode(
            f"verifiers:{self.token}".encode()
        )
        self._callbacks: dict[str, _Callback] = {}
        self._callback_tokens: dict[_Callback, str] = {}
        self._handlers: set[asyncio.Task] = set()
        self.server: asyncio.Server | None = None
        self.port = 0

    def callback_url(
        self,
        url: str,
        host_alias: str = HOST_ALIAS,
        *,
        forward_authorization: bool = True,
    ) -> str:
        """Route one framework-owned host-loopback HTTP(S) origin through this proxy."""
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in ("http", "https") or not is_loopback_host(host):
            raise ValueError(f"unsupported Docker host callback URL: {url}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        authority = parsed.netloc.rpartition("@")[2]
        callback = _Callback(
            parsed.scheme, host, port, authority, host_alias, forward_authorization
        )
        token = self._callback_tokens.get(callback)
        if token is None:
            token = secrets.token_urlsafe(32)
            self._callback_tokens[callback] = token
            self._callbacks[token] = callback
        userinfo, separator, _ = parsed.netloc.rpartition("@")
        netloc = f"{userinfo}{separator}{host_alias}:{self.port}"
        path = f"{_CALLBACK_PREFIX}{token}{parsed.path}"
        return urlunsplit(("http", netloc, path, parsed.query, parsed.fragment))

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
        server, self.server = self.server, None
        server.close()
        # Accepted streams can outlive the listener, especially long-lived callbacks.
        handlers = list(self._handlers)
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        await server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self.server is None:
            writer.close()
            return
        handler = asyncio.current_task()
        assert handler is not None
        self._handlers.add(handler)
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
            method = request.method.decode("ascii")
            target = request.target.decode("ascii")
            parsed = urlsplit(target)
            callback = None
            if method != "CONNECT" and parsed.path.startswith(_CALLBACK_PREFIX):
                token, separator, path = parsed.path[len(_CALLBACK_PREFIX) :].partition(
                    "/"
                )
                callback = self._callbacks.get(token)
                if callback is not None:
                    parsed = parsed._replace(path=f"/{path}" if separator else "/")
            authorization = next(
                (
                    value
                    for name, value in request.headers
                    if name.lower() == b"proxy-authorization"
                ),
                b"",
            )
            if callback is None and not hmac.compare_digest(
                authorization, self._authorization
            ):
                response_started = True
                writer.write(
                    b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                    b'Proxy-Authenticate: Basic realm="verifiers"\r\n'
                    b"Content-Length: 0\r\n\r\n"
                )
                await _drain(writer)
                return
            connect = method == "CONNECT"
            if callback is not None:
                scheme, host, port = callback.scheme, callback.host, callback.port
            elif connect:
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
            permitted = callback is not None or (
                (connect or scheme == "http")
                and self.policy.permits(scheme, host, port, connect=connect)
            )
            addresses = []
            if permitted:
                dial_host = host
                if host.lower() == HOST_ALIAS:
                    dial_host = "127.0.0.1"
                elif callback is not None and host.lower().endswith(".localhost"):
                    dial_host = "localhost"
                addresses = await asyncio.wait_for(
                    asyncio.get_running_loop().getaddrinfo(
                        dial_host, port, type=socket.SOCK_STREAM
                    ),
                    _IO_TIMEOUT,
                )
                framework = any(
                    network_rule_matches(route, scheme, host, port)
                    for route in self.policy.routes
                )
                if (
                    callback is None
                    and not framework
                    and not self.policy.allow_non_global
                ):
                    for *_, address in addresses:
                        resolved = ip_address(address[0])
                        mapped = getattr(resolved, "ipv4_mapped", None)
                        if not (mapped or resolved).is_global:
                            permitted = False
                            break
            if not permitted:
                response_started = True
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await _drain(writer)
                return
            tls = callback is not None and scheme == "https"
            for family, _, _, _, address in addresses:
                try:
                    upstream_reader, upstream_writer = await asyncio.wait_for(
                        asyncio.open_connection(
                            address[0],
                            address[1],
                            family=family,
                            flags=socket.AI_NUMERICHOST,
                            ssl=True if tls else None,
                            server_hostname=host if tls else None,
                        ),
                        _HEADER_TIMEOUT,
                    )
                    break
                except (OSError, TimeoutError):
                    continue
            if upstream_reader is None or upstream_writer is None:
                raise ConnectionError(f"could not connect to {host}:{port}")
            if connect:
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
                read_timeout = None if callback is not None else _IO_TIMEOUT
                path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
                if callback is not None:
                    authority = callback.authority
                else:
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
                if callback is not None and not callback.forward_authorization:
                    excluded.add(b"authorization")
                origin_rewrites = (
                    {
                        f"http://{callback.host_alias}:{self.port}".lower().encode(): f"{scheme}://{callback.authority}".encode()
                    }
                    if callback is not None
                    else {}
                )
                headers = [
                    (
                        name,
                        origin_rewrites.get(value.lower(), value)
                        if name.lower() == b"origin"
                        else value,
                    )
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
                        client.receive_data(await _read(reader, _IO_TIMEOUT))
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
                if callback is not None:
                    while True:
                        head = await asyncio.wait_for(
                            upstream_reader.readuntil(b"\r\n\r\n"), read_timeout
                        )
                        upstream.receive_data(head)
                        response = upstream.next_event()
                        if not isinstance(
                            response, (h11.InformationalResponse, h11.Response)
                        ):
                            raise TypeError("expected an HTTP response")
                        headers = []
                        for name, value in response.headers:
                            if name.lower() == b"location":
                                destination = urljoin(
                                    f"{scheme}://{callback.authority}{path}",
                                    value.decode("latin-1"),
                                )
                                redirected = urlsplit(destination)
                                redirect_host = (
                                    (redirected.hostname or "").lower().rstrip(".")
                                )
                                if redirected.scheme in (
                                    "http",
                                    "https",
                                ) and is_loopback_host(redirect_host):
                                    redirect_port = redirected.port or (
                                        443 if redirected.scheme == "https" else 80
                                    )
                                    value = self.callback_url(
                                        destination,
                                        callback.host_alias,
                                        forward_authorization=(
                                            callback.forward_authorization
                                            and redirected.scheme == scheme
                                            and redirect_host == callback.host
                                            and redirect_port == callback.port
                                        ),
                                    ).encode("latin-1")
                            headers.append((name, value))
                        response = replace(response, headers=headers)
                        response_started = True
                        writer.write(client.send(response))
                        await _drain(writer)
                        if isinstance(response, h11.Response):
                            break
                while chunk := await _read(upstream_reader, read_timeout):
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
            self._handlers.discard(handler)


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
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
