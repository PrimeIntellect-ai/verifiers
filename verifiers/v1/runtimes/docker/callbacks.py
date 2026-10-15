"""In-process HTTP relay that lets a Docker container reach host-loopback services.

A bridged container cannot dial the host's 127.0.0.1. Each host origin a program in the
container needs (interception server, host tool servers) gets a capability URL on this
relay, `http://<alias>:<port>/.vf-host/<token>/<path>`, which forwards to the origin."""

import asyncio
import contextlib
import secrets
import socket
from dataclasses import dataclass, replace
from ipaddress import ip_address
from urllib.parse import urljoin, urlsplit, urlunsplit

import h11

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


@dataclass(frozen=True)
class _Callback:
    scheme: str
    host: str
    port: int
    authority: str
    host_alias: str


class CallbackRelay:
    def __init__(self) -> None:
        self._callbacks: dict[str, _Callback] = {}
        self._callback_tokens: dict[_Callback, str] = {}
        self._handlers: set[asyncio.Task] = set()
        self.server: asyncio.Server | None = None
        self.port = 0

    def callback_url(self, url: str, host_alias: str) -> str:
        """Route one framework-owned host-loopback HTTP(S)/WebSocket origin."""
        parsed = urlsplit(url)
        scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
        host = (parsed.hostname or "").lower().rstrip(".")
        if scheme not in ("http", "https") or not is_loopback_host(host):
            raise ValueError(f"unsupported Docker host callback URL: {url}")
        port = parsed.port or (443 if scheme == "https" else 80)
        authority = parsed.netloc.rpartition("@")[2]
        callback = _Callback(scheme, host, port, authority, host_alias)
        token = self._callback_tokens.get(callback)
        if token is None:
            token = secrets.token_urlsafe(32)
            self._callback_tokens[callback] = token
            self._callbacks[token] = callback
        userinfo, separator, _ = parsed.netloc.rpartition("@")
        netloc = f"{userinfo}{separator}{host_alias}:{self.port}"
        path = f"{_CALLBACK_PREFIX}{token}{parsed.path or '/'}"
        visible_scheme = "ws" if parsed.scheme in ("ws", "wss") else "http"
        return urlunsplit((visible_scheme, netloc, path, parsed.query, parsed.fragment))

    async def start(
        self, bind_host: str | None = None, *, listener: socket.socket | None = None
    ) -> None:
        if listener is None:
            self.server = await asyncio.start_server(self._relay, bind_host, 0)
        else:
            self.server = await asyncio.start_server(self._relay, sock=listener)
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

    async def _relay(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self.server is None:
            writer.close()
            return
        handler = asyncio.current_task()
        assert handler is not None
        self._handlers.add(handler)
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
            parsed = urlsplit(request.target.decode("ascii"))
            callback = None
            if parsed.path.startswith(_CALLBACK_PREFIX):
                token, separator, path = parsed.path[len(_CALLBACK_PREFIX) :].partition(
                    "/"
                )
                callback = self._callbacks.get(token)
                parsed = parsed._replace(path=f"/{path}" if separator else "/")
            if request.method == b"CONNECT" or callback is None:
                response_started = True
                writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await _drain(writer)
                return
            scheme, host, port = callback.scheme, callback.host, callback.port
            dial_host = "localhost" if host.endswith(".localhost") else host
            addresses = await asyncio.wait_for(
                asyncio.get_running_loop().getaddrinfo(
                    dial_host, port, type=socket.SOCK_STREAM
                ),
                _IO_TIMEOUT,
            )
            tls = scheme == "https"
            upstream_reader = None
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
            path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            connection_fields = {
                field.strip().lower()
                for name, value in request.headers
                if name.lower() == b"connection"
                for field in value.split(b",")
            }
            websocket = b"upgrade" in connection_fields and any(
                name.lower() == b"upgrade" and value.lower() == b"websocket"
                for name, value in request.headers
            )
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
            if websocket:
                excluded.discard(b"upgrade")
            relay_origin = f"http://{callback.host_alias}:{self.port}".lower().encode()
            headers = []
            for name, value in request.headers:
                if name.lower() in excluded:
                    continue
                if name.lower() == b"cookie":
                    # Callback cookie names carry their capability token; strip it upstream.
                    prefix = f"{token}-".encode()
                    value = b"; ".join(
                        cookie.strip().removeprefix(prefix)
                        for cookie in value.split(b";")
                    )
                    if not value:
                        continue
                if name.lower() == b"origin" and value.lower() == relay_origin:
                    value = f"{scheme}://{callback.authority}".encode()
                headers.append((name, value))
            upstream = h11.Connection(h11.CLIENT)
            upstream_writer.write(
                upstream.send(
                    h11.Request(
                        method=request.method,
                        target=path,
                        headers=[
                            (b"Host", callback.authority.encode("ascii")),
                            (b"Connection", b"Upgrade" if websocket else b"close"),
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
                    client.send(h11.InformationalResponse(status_code=100, headers=[]))
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
            while True:
                head = await upstream_reader.readuntil(b"\r\n\r\n")
                upstream.receive_data(head)
                response = upstream.next_event()
                if not isinstance(response, (h11.InformationalResponse, h11.Response)):
                    raise TypeError("expected an HTTP response")
                headers = []
                for name, value in response.headers:
                    if name.lower() == b"set-cookie":
                        value = self._scope_cookie(value, token, scheme, parsed.path)
                    if name.lower() == b"location":
                        value = self._relay_location(value, callback, path)
                    headers.append((name, value))
                response = replace(response, headers=headers, http_version=b"1.1")
                response_started = True
                writer.write(client.send(response))
                await _drain(writer)
                if response.status_code == 101:
                    await _pipe_both(reader, writer, upstream_reader, upstream_writer)
                    return
                if isinstance(response, h11.Response):
                    break
            # h11 may change framing, so serialize the body through it too.
            while True:
                event = upstream.next_event()
                if event is h11.NEED_DATA:
                    upstream.receive_data(await _read(upstream_reader, None))
                    continue
                writer.write(client.send(event))
                await _drain(writer)
                if isinstance(event, h11.EndOfMessage):
                    return
        except Exception:  # noqa: BLE001 - relay failures become a generic 502
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

    @staticmethod
    def _scope_cookie(value: bytes, token: str, scheme: str, path: str) -> bytes:
        """Cookie jars see the local HTTP hop. Scope cookies to this capability,
        preserving unknown cookie attributes."""
        cookie, *attributes = value.split(b";")
        parts, scope = [f"{token}-".encode() + cookie], b""
        for attribute in attributes:
            key, _, content = attribute.strip().partition(b"=")
            if key.lower() == b"path":
                scope = content
            elif key.lower() != b"domain" and not (
                scheme == "https"
                and (
                    key.lower() in (b"secure", b"partitioned")
                    or (
                        key.lower() == b"samesite"
                        and content.strip().lower() == b"none"
                    )
                )
            ):
                parts.append(attribute)
        if not scope.startswith(b"/"):
            scope = path.rpartition("/")[0].encode() or b"/"
        parts.append(f" Path={_CALLBACK_PREFIX}{token}".encode() + scope)
        return b";".join(parts)

    def _relay_location(self, value: bytes, callback: _Callback, path: str) -> bytes:
        destination = urljoin(
            f"{callback.scheme}://{callback.authority}{path}", value.decode("latin-1")
        )
        redirected = urlsplit(destination)
        target = (
            redirected.scheme,
            (redirected.hostname or "").lower().rstrip("."),
            redirected.port or (443 if redirected.scheme == "https" else 80),
        )
        if target != (callback.scheme, callback.host, callback.port):
            return destination.encode("latin-1")
        # Redirects cannot grant access to another host service.
        return self.callback_url(
            redirected._replace(netloc=callback.authority).geturl(),
            callback.host_alias,
        ).encode("latin-1")


async def _pipe_both(
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
