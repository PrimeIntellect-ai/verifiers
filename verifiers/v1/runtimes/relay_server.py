# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Loopback HTTP relay between a harness program in the box and the host's interception
server.

The harness is launched against `http://127.0.0.1:<port>` instead of the host's URL. The
relay forwards every request to the configured upstream and swaps the box's bearer
tokens for the upstream's. Pausing and resuming the box keeps the harness's loopback
connections but breaks the relay's upstream ones, and the resumed box is served at a new
URL with new secrets. On SIGHUP the relay re-reads its config, aborts every in-flight
upstream request and retries each one against the new upstream, as long as its client
has seen nothing but the status, headers and SSE comments: a model call still waiting
for its first event.

Usage: relay_server.py --config <path> --port <port> [--ready <path>] [--pid <path>]

The config file is JSON: `{"upstream": "https://host/base", "tokens": {"<box token>":
"<upstream token>"}}`. Once listening, the relay writes its port to `--ready` and its
process id to `--pid`.
"""

import argparse
import contextlib
import http.client
import json
import os
import signal
import socket
import ssl
import struct
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

KEEPALIVE_SECONDS = 10.0
"""Interval of the comments an SSE client gets while its request waits for a retry."""
RECONNECT_SECONDS = 15 * 60
"""How long a failing upstream connect is retried once a SIGHUP happened: the host may
rewrite the config before the new upstream is up."""
CONNECT_ATTEMPTS = 3
"""Connect attempts before the first SIGHUP: a failing connect then means the host is
down, and the harness must learn that fast."""
BACKOFF_SECONDS = 0.5
BACKOFF_CAP_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 10.0
CHUNK_BYTES = 64 * 1024
RETRY_COUNT_HEADER = "x-stainless-retry-count"
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
# The relay sends the whole body itself and lets http.client set Host for the upstream.
REQUEST_DROPPED = HOP_BY_HOP | {"host", "content-length", "expect"}
# Not terminated by a blank line, like the interception server's own keepalives: some
# SSE clients try to decode the empty event a blank line would dispatch.
KEEPALIVE = b": relay keepalive\n"
STATUS_CHANGED = "relay: upstream changed status on retry"
SSL_CONTEXT = ssl.create_default_context()


def log(message: str) -> None:
    print(f"relay: {message}", file=sys.stderr, flush=True)


def error_event(message: str) -> bytes:
    return b"event: error\ndata: " + json.dumps({"error": message}).encode() + b"\n\n"


class Config:
    def __init__(self, upstream: str, tokens: dict[str, str]) -> None:
        parts = urlsplit(upstream)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"upstream must be an http(s) URL: {upstream!r}")
        if not isinstance(tokens, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in tokens.items()
        ):
            raise ValueError("tokens must map strings to strings")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port
        self.prefix = parts.path.rstrip("/")
        self.tokens = tokens
        port = f":{parts.port}" if parts.port else ""
        self.origin = f"{parts.scheme}://{parts.hostname}{port}{self.prefix}"

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as file:
            data = json.load(file)
        return cls(data["upstream"], data.get("tokens") or {})

    def connection(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=CONNECT_TIMEOUT_SECONDS,
                context=SSL_CONTEXT,
            )
        return http.client.HTTPConnection(
            self.host, self.port, timeout=CONNECT_TIMEOUT_SECONDS
        )

    def swap(self, name: str, value: str) -> str:
        if name == "x-api-key":
            return self.tokens.get(value, value)
        scheme, _, token = value.partition(" ")
        token = token.strip()
        if scheme.lower() == "bearer" and token in self.tokens:
            return f"{scheme} {self.tokens[token]}"
        return value


class Upstream:
    """One in-flight upstream attempt, abortable from another thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sock: socket.socket | None = None
        self.aborted = False

    def attach(self, sock: socket.socket) -> None:
        with self.lock:
            self.sock = sock
            aborted = self.aborted
        if aborted:
            shutdown(sock)

    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            sock = self.sock
        if sock is not None:
            shutdown(sock)


def shutdown(sock: socket.socket) -> None:
    # Unlike `close`, `shutdown` wakes a read blocked in another thread. The base class
    # method because `SSLSocket.shutdown` also drops the TLS state that read still uses.
    with contextlib.suppress(OSError):
        socket.socket.shutdown(sock, socket.SHUT_RDWR)


class Relay:
    def __init__(self, path: str) -> None:
        self.path = path
        self.config = Config.load(path)
        self.reloaded = False
        self.changed = threading.Condition()
        self.upstreams: set[Upstream] = set()

    def begin(self) -> tuple[Upstream, Config]:
        """Register an attempt on the current config, atomically with reading it, so a
        reload aborts every attempt that still uses the config it replaces."""
        with self.changed:
            upstream = Upstream()
            self.upstreams.add(upstream)
            return upstream, self.config

    def end(self, upstream: Upstream) -> None:
        with self.changed:
            self.upstreams.discard(upstream)

    def wait(self, config: Config, timeout: float) -> None:
        with self.changed:
            self.changed.wait_for(lambda: self.config is not config, timeout)

    def reload(self) -> None:
        try:
            config = Config.load(self.path)
        except (OSError, ValueError, KeyError) as error:
            # Aborting would only retry against the old upstream.
            log(f"reload failed, keeping {self.config.origin}: {error}")
            return
        with self.changed:
            self.config = config
            self.reloaded = True
            aborted, self.upstreams = self.upstreams, set()
            self.changed.notify_all()
        for upstream in aborted:
            upstream.abort()
        log(f"reloaded: upstream {config.origin}, aborted {len(aborted)}")


class UpstreamError(Exception):
    pass


class ClientGone(Exception):
    pass


class Scanner:
    """Whether a response body forwarded anything but SSE comment lines and blank lines.
    A client skips those, so a retry's body can still follow them."""

    def __init__(self) -> None:
        self.sse = False
        self.data = False
        self.line_start = True

    def feed(self, chunk: bytes) -> None:
        if self.data or not chunk:
            return
        if not self.sse:
            self.data = True
            return
        for byte in chunk:
            if self.line_start:
                if byte in b"\r\n":
                    continue
                if byte != ord(":"):
                    self.data = True
                    return
                self.line_start = False
            elif byte in b"\r\n":
                self.line_start = True

    def line_break(self) -> bytes:
        """What ends a comment line that a broken upstream left unfinished."""
        return b"" if self.line_start else b"\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "Server"

    def __getattr__(self, name: str):
        # Every method is relayed: `BaseHTTPRequestHandler` dispatches to `do_<METHOD>`.
        if name.startswith("do_"):
            return self.relay
        raise AttributeError(name)

    def log_message(self, format: str, *args) -> None:
        log(format % args)

    def relay(self) -> None:
        try:
            self.body = self.read_body()
        except (OSError, ValueError):
            self.send_error(400, "relay: unreadable request body")
            return
        target = self.path
        if not target.startswith("/"):
            parts = urlsplit(target)
            target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        self.target = target
        dropped = REQUEST_DROPPED | {
            name.strip().lower()
            for name in self.headers.get("Connection", "").split(",")
        }
        self.forwarded = [
            (name, value)
            for name, value in self.headers.items()
            if name.lower() not in dropped
        ]
        try:
            self.retry_count = int(self.headers.get(RETRY_COUNT_HEADER, 0))
        except ValueError:
            self.retry_count = 0
        self.retries = 0
        self.sent_status: int | None = None
        self.sent_length: int | None = None
        self.chunked = False
        self.scanner = Scanner()
        self.write_lock = threading.Lock()
        relay = self.server.relay
        try:
            while True:
                try:
                    with self.keepalive():
                        upstream, connection, response = self.open()
                except UpstreamError as error:
                    log(f"{self.command} {self.label}: {error}")
                    self.fail(str(error))
                    return
                try:
                    done = self.stream(upstream, response)
                finally:
                    relay.end(upstream)
                    connection.close()
                if done:
                    retried = f" after {self.retries} retries" if self.retries else ""
                    log(f"{self.command} {self.label} {self.sent_status}{retried}")
                    return
                if self.scanner.data:
                    log(f"{self.command} {self.label}: aborted after data, closing")
                    self.reset()
                    return
                self.retries += 1
                log(f"{self.command} {self.label}: aborted, retrying")
        except ClientGone:
            log(f"{self.command} {self.label}: client went away")
            self.close_connection = True

    @property
    def label(self) -> str:
        return self.target.partition("?")[0]

    def read_body(self) -> bytes:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks = []
            while True:
                size = int(self.rfile.readline(65537).split(b";")[0], 16)
                if size == 0:
                    while self.rfile.readline(65537) not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline(65537)
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def upstream_headers(self, config: Config) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, value in self.forwarded:
            lower = name.lower()
            if lower in ("authorization", "x-api-key"):
                value = config.swap(lower, value)
            elif lower == RETRY_COUNT_HEADER and self.retries:
                continue
            headers[name] = f"{headers[name]}, {value}" if name in headers else value
        if self.retries:
            headers[RETRY_COUNT_HEADER] = str(self.retry_count + self.retries)
        return headers

    def open(
        self,
    ) -> tuple[Upstream, http.client.HTTPConnection, http.client.HTTPResponse]:
        relay = self.server.relay
        failures = 0
        first_failure = 0.0
        delay = BACKOFF_SECONDS
        while True:
            upstream, config = relay.begin()
            connection = config.connection()
            connected = False
            try:
                connection.connect()
                assert connection.sock is not None
                upstream.attach(connection.sock)
                # Only the connect is bounded: a model call may wait long for its first
                # byte.
                connection.sock.settimeout(None)
                connected = True
                connection.request(
                    self.command,
                    config.prefix + self.target,
                    self.body or None,
                    self.upstream_headers(config),
                )
                return upstream, connection, connection.getresponse()
            except (OSError, http.client.HTTPException) as error:
                relay.end(upstream)
                connection.close()
                if upstream.aborted:
                    self.retries += 1
                    log(f"{self.command} {self.label}: aborted, retrying")
                    continue
                if connected:
                    raise UpstreamError(
                        f"upstream request failed: {error!r}"
                    ) from error
                failures += 1
                now = time.monotonic()
                first_failure = first_failure or now
                if (
                    now - first_failure >= RECONNECT_SECONDS
                    if relay.reloaded
                    else failures >= CONNECT_ATTEMPTS
                ):
                    raise UpstreamError(
                        f"upstream connect failed: {error!r}"
                    ) from error
                log(f"connect to {config.origin} failed ({error!r}), retry in {delay}s")
                relay.wait(config, delay)
                delay = min(delay * 2, BACKOFF_CAP_SECONDS)

    @contextlib.contextmanager
    def keepalive(self) -> Iterator[None]:
        """Keep an SSE client's idle timer from firing while its request waits for a
        retry."""
        if self.sent_status is None or not self.scanner.sse:
            yield
            return
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(KEEPALIVE_SECONDS):
                try:
                    self.send(self.scanner.line_break() + KEEPALIVE)
                except ClientGone:
                    return

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def stream(self, upstream: Upstream, response: http.client.HTTPResponse) -> bool:
        """Forward one upstream response; False when a reload aborted it."""
        if self.sent_status is None:
            self.start(response)
        elif response.status != self.sent_status:
            log(
                f"{self.command} {self.label}: retry answered {response.status}, "
                f"sent {self.sent_status}"
            )
            if self.scanner.sse:
                self.send(self.scanner.line_break() + error_event(STATUS_CHANGED))
                self.finish_body()
            else:
                self.reset()
            return True
        elif self.sent_length is not None and response.length != self.sent_length:
            self.reset()
            return True
        else:
            self.send(self.scanner.line_break())
        while True:
            try:
                data = response.read1(CHUNK_BYTES)
            except (OSError, http.client.HTTPException) as error:
                if upstream.aborted:
                    return False
                log(f"{self.command} {self.label}: upstream broke: {error!r}")
                self.reset()
                return True
            # Data read after the abort is dropped: the retry sends it again.
            if upstream.aborted:
                return False
            if not data:
                self.finish_body()
                return True
            self.send(data)

    def start(self, response: http.client.HTTPResponse) -> None:
        status = response.status
        content_type = response.getheader("Content-Type", "")
        encoding = response.getheader("Content-Encoding", "identity")
        self.scanner.sse = content_type.startswith("text/event-stream") and (
            encoding.lower() == "identity"
        )
        dropped = HOP_BY_HOP | {"content-length"}
        dropped |= {
            name.strip().lower()
            for name in (response.getheader("Connection") or "").split(",")
        }
        self.send_response_only(status, response.reason)
        for name, value in response.getheaders():
            if name.lower() not in dropped:
                self.send_header(name, value)
        bodyless = self.command == "HEAD" or status in (204, 304) or status < 200
        if bodyless:
            if length := response.getheader("Content-Length"):
                self.send_header("Content-Length", length)
        elif response.length is not None and not response.chunked:
            self.sent_length = response.length
            self.send_header("Content-Length", str(response.length))
        elif self.request_version != "HTTP/1.0":
            self.chunked = True
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.close_connection = True
            self.send_header("Connection", "close")
        try:
            self.end_headers()
        except OSError as error:
            raise ClientGone from error
        self.sent_status = status

    def send(self, data: bytes) -> None:
        if not data:
            return
        with self.write_lock:
            self.scanner.feed(data)
            if self.chunked:
                data = b"%x\r\n%s\r\n" % (len(data), data)
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except OSError as error:
                raise ClientGone from error

    def finish_body(self) -> None:
        if self.chunked:
            with self.write_lock:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError as error:
                    raise ClientGone from error

    def fail(self, message: str) -> None:
        if self.sent_status is None:
            body = json.dumps({"error": f"relay: {message}"}).encode()
            self.send_response_only(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            try:
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)
            except OSError as error:
                raise ClientGone from error
        elif self.scanner.sse:
            self.send(self.scanner.line_break() + error_event(f"relay: {message}"))
            self.finish_body()
        else:
            self.reset()

    def reset(self) -> None:
        """Break the client's connection so it cannot take a cut body for a whole one."""
        self.close_connection = True
        with contextlib.suppress(OSError):
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], relay: Relay) -> None:
        self.relay = relay
        super().__init__(address, Handler)


def write_file(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as file:
        file.write(text)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready")
    parser.add_argument("--pid")
    args = parser.parse_args()
    relay = Relay(args.config)
    # Blocked before any thread starts, so threads inherit the mask and only the
    # `sigwait` below takes SIGHUP: a reload then never interrupts a handler.
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGHUP})
    server = Server(("127.0.0.1", args.port), relay)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if args.pid:
        write_file(args.pid, f"{os.getpid()}\n")
    if args.ready:
        write_file(args.ready, f"{port}\n")
    log(f"listening on 127.0.0.1:{port}, upstream {relay.config.origin}")
    while True:
        signal.sigwait({signal.SIGHUP})
        relay.reload()


if __name__ == "__main__":
    main()
