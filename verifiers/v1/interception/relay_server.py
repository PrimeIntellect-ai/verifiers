"""Loopback relay from a remote runtime's harness to the host's model endpoint.

python3 -I -S relay.pyz UPSTREAM FILES WINDOW_SECONDS

Runs inside the runtime from a zip that bundles h11 (see `relay.py`). Each request goes to
UPSTREAM on a fresh connection and its response streams back. The host stamps every
response, so a 404, 408, 429 or 5xx without the stamp came from a tunnel or proxy in
between: those and failed connections are retried for up to WINDOW_SECONDS, marked so the
host answers a repeat with the original result. Then the relay gives up as a direct
connection would: it forwards the last answer, or closes. FILES is a path prefix for
FILES.port (the port, or why it refused) and FILES.rescued (a byte per rescued request).
"""

from __future__ import annotations

import os
import random
import select
import socket
import socketserver
import ssl
import struct
import sys
import threading
import time
from urllib.parse import urlsplit

import h11

STAMP = b"x-verifiers-interception"
RETRY = b"x-stainless-retry-count"
HOP_BY_HOP = {
    b"connection",
    b"expect",
    b"host",
    b"keep-alive",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
MAX_BODY = 1 << 30  # the host's own limit


def transient(response: h11.Response) -> bool:
    status = response.status_code
    stamped = any(name == STAMP for name, _ in response.headers)
    return not stamped and (status in (404, 408, 429) or status >= 500)


def receive(conn: h11.Connection, sock: socket.socket):
    while True:
        event = conn.next_event()
        if event is not h11.NEED_DATA:
            return event
        conn.receive_data(sock.recv(1 << 16))


class Upstream:
    def __init__(self, url: str) -> None:
        url = urlsplit(url)
        self.hostname, self.port = url.hostname, url.port
        self.host = url.netloc.rpartition("@")[2].encode()
        self.base = url.path.rstrip("/").encode()
        self.tls = ssl.create_default_context() if url.scheme == "https" else None
        self.port = self.port or (443 if self.tls else 80)

    def send(
        self,
        method: bytes,
        target: bytes,
        headers: list,
        body: bytes,
        timeout: float | None = None,
    ):
        """Send one request on a fresh connection: (socket, connection, response).
        Raises OSError or h11.ProtocolError if it got no response."""
        sock = socket.create_connection((self.hostname, self.port), timeout=30)
        try:
            if self.tls:
                sock = self.tls.wrap_socket(sock, server_hostname=self.hostname)
            # Model turns can stream for many minutes, so no read timeout. Keepalive
            # probes, and a cap on unacknowledged sends, notice a connection that died.
            sock.settimeout(timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for option, value in (
                ("TCP_KEEPIDLE", 30),
                ("TCP_KEEPINTVL", 10),
                ("TCP_KEEPCNT", 6),
                ("TCP_USER_TIMEOUT", 90_000),
            ):
                if hasattr(socket, option):
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)
            conn = h11.Connection(h11.CLIENT)
            headers = [(b"host", self.host), *headers, (b"connection", b"close")]
            headers.append((b"content-length", b"%d" % len(body)))
            request = h11.Request(
                method=method, target=self.base + target, headers=headers
            )
            try:
                sock.sendall(conn.send(request))
                for data in conn.send_with_data_passthrough(h11.Data(data=body)):
                    sock.sendall(data)
                sock.sendall(conn.send(h11.EndOfMessage()))
            except OSError:
                pass  # it may have answered (say, a 413) before taking it all
            event = receive(conn, sock)
            while isinstance(event, h11.InformationalResponse):
                event = receive(conn, sock)
            if not isinstance(event, h11.Response):
                raise h11.RemoteProtocolError(f"no response: {event!r}")
            return sock, conn, event
        except BaseException:
            sock.close()
            raise

    def check(self) -> str:
        """Why requests can't get through this way: "" if they can, or if the network
        is just down (retries handle that)."""
        try:
            sock, _, response = self.send(b"GET", b"/v1/models", [], b"", timeout=30)
            sock.close()
        except ssl.SSLError as e:
            dropped = isinstance(e, (ssl.SSLEOFError, ssl.SSLZeroReturnError))
            return "" if dropped else repr(e)  # say, no CA certificates
        except (OSError, h11.ProtocolError):
            return ""
        stamped = any(name == STAMP for name, _ in response.headers)
        if stamped or transient(response):
            return ""
        return f"its address answered {response.status_code} without the host's stamp"


class Relay(socketserver.BaseRequestHandler):
    server: Server

    def handle(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client = h11.Connection(h11.SERVER, max_incomplete_event_size=1 << 20)
        try:
            while self.relay():
                self.client.start_next_cycle()
        except (OSError, h11.ProtocolError):
            pass  # a malformed request, or a client or host that left mid-message

    def relay(self) -> bool:
        """Relay one request; whether the connection can take another."""
        request = receive(self.client, self.request)
        if not isinstance(request, h11.Request):
            return False
        if self.client.they_are_waiting_for_100_continue:
            self.respond(h11.InformationalResponse(status_code=100, headers=[]))
        body = bytearray()
        while isinstance(event := receive(self.client, self.request), h11.Data):
            body += event.data
            if len(body) > MAX_BODY:
                return False
        if not isinstance(event, h11.EndOfMessage):
            return False  # the client left mid-body
        body = bytes(body)
        if not request.target.startswith(b"/v1/"):
            self.respond(h11.Response(status_code=404, headers=[]))
            self.respond(h11.EndOfMessage())
            return self.client.our_state is h11.DONE
        connection = {
            token.strip().lower()
            for name, value in request.headers
            if name == b"connection"
            for token in value.split(b",")
        }
        headers = [
            (name, value)
            for name, value in request.headers
            if name not in HOP_BY_HOP | connection | {b"content-length"}
        ]
        upstream: Upstream = self.server.upstream
        held, delay, retry = 0.0, 0.5, 0
        while True:
            try:
                sock, conn, response = upstream.send(
                    request.method, request.target, headers, body
                )
                if not transient(response):
                    break
            except (OSError, h11.ProtocolError):
                sock = response = None
            held = held or time.monotonic()
            left = held + self.server.window - time.monotonic()
            if left <= 0:
                break  # as a direct connection would: the last answer, or none
            if sock is not None:
                sock.close()
            if self.gone(min(delay * random.uniform(0.5, 1.5), left)):
                return False
            delay, retry = min(delay * 2, 10.0), retry + 1
            headers = [(k, v) for k, v in headers if k != RETRY] + [
                (RETRY, b"%d" % retry)
            ]
        if response is None:
            return False
        if retry and not transient(response):
            self.server.rescued()
        try:
            self.forward(conn, sock, response)
        finally:
            sock.close()
        return self.client.our_state is h11.DONE

    def forward(self, conn: h11.Connection, sock: socket.socket, response) -> None:
        # h11 frames the body for this client: kept Content-Length, else chunked.
        headers = [(k, v) for k, v in response.headers if k not in HOP_BY_HOP]
        self.respond(
            h11.Response(
                status_code=response.status_code,
                headers=headers,
                reason=response.reason,
            )
        )
        try:
            while isinstance(event := receive(conn, sock), h11.Data):
                self.respond(h11.Data(data=event.data))
            if not isinstance(event, h11.EndOfMessage):
                raise h11.RemoteProtocolError(f"body ended early: {event!r}")
            self.respond(h11.EndOfMessage())
        except BaseException:
            # A body cut short must not look complete to the client: reset the
            # connection (now, before the server's graceful shutdown), which even a
            # client reading to the end of it notices.
            self.request.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.request.close()
            raise

    def respond(self, event) -> None:
        self.request.sendall(self.client.send(event))

    def gone(self, seconds: float) -> bool:
        """Wait between attempts; whether the client hung up meanwhile."""
        poll = select.poll()
        poll.register(self.request, getattr(select, "POLLRDHUP", 0))
        return bool(poll.poll(seconds * 1000))  # hang-ups and errors, not new requests


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024

    def __init__(self, port: int, upstream: str, files: str, window: float) -> None:
        super().__init__(("127.0.0.1", port), Relay)
        self.upstream, self.files, self.window = Upstream(upstream), files, window
        self.lock = threading.Lock()
        open(f"{files}.rescued", "ab").close()

    def rescued(self) -> None:
        with self.lock, open(f"{self.files}.rescued", "ab") as f:
            f.write(b"1")


def main() -> None:
    upstream, files, window = sys.argv[1:4]
    try:
        with open(f"{files}.port") as f:
            port = int(f.read())  # a restart keeps the address
    except (OSError, ValueError):
        port = 0
    server = Server(port, upstream, files, float(window))
    # Checked on first start; a restart serves at once, while connections queue.
    refused = "" if port else server.upstream.check()
    with open(f"{files}.port.tmp", "w") as f:
        f.write(
            f"refused: {ascii(refused)[:400]}"
            if refused
            else str(port or server.server_address[1])
        )
    os.replace(f"{files}.port.tmp", f"{files}.port")
    if refused:
        threading.Event().wait()  # the host goes direct and stops this
    server.serve_forever()


if __name__ == "__main__":
    main()
