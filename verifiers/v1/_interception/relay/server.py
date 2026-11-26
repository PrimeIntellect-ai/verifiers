# /// script
# requires-python = ">=3.10"
# dependencies = ["certifi"]
# ///
"""Loopback HTTP relay between a harness program in the box and the host's interception
server.

The harness is launched against `http://127.0.0.1:<port>` instead of the host's URL. The
relay forwards every request to the configured upstream and swaps the box's bearer
tokens for the upstream's. The upstream can change while the harness runs: the host
replaces a lost tunnel with a new URL. On SIGHUP the relay re-reads its config, aborts
every in-flight upstream request and retries each one against the new upstream, as
long as its client has seen nothing but the status, headers and SSE comments: a model
call still waiting for its first event.

While the upstream cannot be reached (a connect fails, a request breaks before its
response, the tunnel edge answers with its own error page), the relay retries the
request for up to `RETRY_SECONDS`, so the harness does not see a tunnel that is
reconnecting or being replaced. A retry is marked as one (`x-stainless-retry-count`),
so the interception server joins it to the request it already serves.

Usage: relay_server.py --config <path> --port <port> [--ready <path>] [--pid <path>]
                       [--failures <path>]

The config file is JSON: `{"upstream": "https://host/base", "tokens": {"<box token>":
"<upstream token>"}}`. Once listening, the relay writes its port to `--ready` and its
process id to `--pid`. Each request it gives up on appends its reason to `--failures`,
so the host can tell a failure of the relay from one of the harness.
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
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import certifi

KEEPALIVE_SECONDS = 10.0
"""Interval of the comments an SSE client gets while its request waits for a retry."""
RETRY_SECONDS = 5 * 60
"""How long a request is retried while the upstream cannot be reached: long enough for
the host to notice a lost tunnel, open a new one and reroute the relay."""
EDGE_STATUSES = frozenset({404, 502, 503, 504})
"""Statuses a tunnel edge answers with its own page when it cannot reach the upstream
("404 Tunnel Not Found", a bad gateway)."""
SERVER_HEADER = "x-vf-interception"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
"""Header on every response of the interception server, its own 404 for an unknown path
included. A response with an edge status and without it never reached the server."""
BACKOFF_SECONDS = 0.5
BACKOFF_CAP_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 10.0
SILENT_UPSTREAM_SECONDS = 90.0
"""How long a retry may wait for its response while the relay keeps the client alive.
The client's own idle timeout cannot fire then, and the interception server answers a
streamed request within a minute (headers, then a keepalive every few seconds), so a
longer silence is a dead connection, not a slow model call."""
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
STATUS_CHANGED = "upstream changed status on retry"
# certifi's roots, as the harness's own HTTP clients use: a task image may have no system
# CA store (no `ca-certificates`), and the stdlib then trusts no certificate at all.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SSL_CONTEXT.load_default_certs()


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
    def __init__(self, path: str, failures: str | None = None) -> None:
        self.path = path
        self.failures = failures
        self.failures_lock = threading.Lock()
        self.failed = False
        """Whether the last request outcome recorded in `failures` is a failure."""
        self.config = Config.load(path)
        self.changed = threading.Condition()
        self.upstreams: set[Upstream] = set()

    def begin(self) -> tuple[Upstream, Config]:
        """Register an attempt on the current config, atomically with reading it, so a
        reload aborts every attempt that still uses the config it replaces."""
        with self.changed:
            upstream = Upstream()
            self.upstreams.add(upstream)
            return upstream, self.config

    def record_failure(self, reason: str) -> None:
        self._record(reason.replace("\n", " "), failed=True)

    def record_success(self) -> None:
        """After a failure, record that a request went through again: the failure is
        no longer the relay's current state."""
        if self.failed:
            self._record("ok", failed=False)

    def _record(self, line: str, *, failed: bool) -> None:
        if self.failures is None:
            return
        with (
            self.failures_lock,
            contextlib.suppress(OSError),
            open(self.failures, "a") as file,
        ):
            file.write(line + "\n")
            self.failed = failed

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
            aborted, self.upstreams = self.upstreams, set()
            self.changed.notify_all()
        for upstream in aborted:
            upstream.abort()
        log(f"reloaded: upstream {config.origin}, aborted {len(aborted)}")


class UpstreamError(Exception):
    pass


def is_edge_error(response: http.client.HTTPResponse) -> bool:
    """Whether `response` is the tunnel edge's own error page, not the upstream's answer."""
    return (
        response.status in EDGE_STATUSES and response.getheader(SERVER_HEADER) is None
    )


class ClientGone(Exception):
    pass


# The events the interception server keeps a waiting stream alive with, for clients that
# only count events (each dialect's `stream_keepalive`): a retry may send them again.
KEEPALIVE_EVENTS = (b"ping", b"response.created", b"response.in_progress")
KEEPALIVE_DATA = (b'"type": "ping"', b'"id": "resp_keepalive"')


class Scanner:
    """Whether a response body forwarded anything but what a retry's body can still
    follow: SSE comment lines, blank lines, and the server's keepalive events. A
    line not yet complete counts as data unless it is a comment, and so does an
    event until its data line shows it is a keepalive: real events share the names."""

    def __init__(self) -> None:
        self.sse = False
        self.sent_data = False
        self.line = bytearray()
        self.in_event = False
        """A field line came since the last blank line: the event is still open."""
        self.keepalive = False
        """The open event's name is a keepalive's."""
        self.confirmed = False
        """The open event's data is a keepalive's."""

    @property
    def data(self) -> bool:
        if self.sent_data or (self.in_event and not self.confirmed):
            return True
        return bool(self.line) and not self.line.startswith(b":")

    def feed(self, chunk: bytes) -> None:
        if self.sent_data or not chunk:
            return
        if not self.sse:
            self.sent_data = True
            return
        for byte in chunk:
            if byte != ord("\n"):
                self.line.append(byte)
                continue
            self.end_line(bytes(self.line).rstrip(b"\r"))
            self.line.clear()
            if self.sent_data:
                return

    def end_line(self, line: bytes) -> None:
        if not line:
            self.in_event = self.keepalive = self.confirmed = False
            return
        if line.startswith(b":"):
            return
        field, _, value = line.partition(b":")
        value = value.lstrip(b" ")
        if not self.in_event:
            self.in_event = True
            self.keepalive = field == b"event" and value in KEEPALIVE_EVENTS
        if field == b"event":
            ok = self.keepalive and value in KEEPALIVE_EVENTS
        elif field == b"data":
            ok = self.keepalive and any(mark in value for mark in KEEPALIVE_DATA)
            self.confirmed = ok
        else:
            ok = False
        if not ok:
            self.sent_data = True

    def line_break(self) -> bytes:
        """What ends a comment line and a keepalive event that a broken upstream left
        unfinished."""
        return (b"\n" if self.line else b"") + (b"\n" if self.in_event else b"")


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
        if self.headers.get(IDEMPOTENCY_KEY_HEADER) is None:
            # Every attempt of this request carries the key, so the interception server
            # answers one that an edge page hid from the relay instead of sampling again.
            self.forwarded.append((IDEMPOTENCY_KEY_HEADER, uuid.uuid4().hex))
        try:
            self.retry_count = int(self.headers.get(RETRY_COUNT_HEADER, 0))
        except ValueError:
            self.retry_count = 0
        self.retries = 0
        self.sent = 0
        """Attempts that may have reached the interception server: the retry count it
        sees. An attempt that failed to connect or got the tunnel edge's own page is
        not counted, so a request that a legitimate later turn repeats is never taken
        for a retry and answered from the server's replay cache. An edge page can still
        hide an attempt that reached the server: the request's idempotency key covers
        that one."""
        self.failing_since: float | None = None
        self.delay = BACKOFF_SECONDS
        self.sent_status: int | None = None
        self.sent_length: int | None = None
        self.chunked = False
        self.scanner = Scanner()
        self.gave_up = False
        """Whether the relay gave up on this request (`fail`, `abandon`): it then never
        counts as one that went through."""
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
                opened = time.monotonic()
                try:
                    done = self.stream(upstream, response)
                finally:
                    relay.end(upstream)
                    connection.close()
                if done:
                    if not self.gave_up:
                        relay.record_success()
                        retried = (
                            f" after {self.retries} retries" if self.retries else ""
                        )
                        log(f"{self.command} {self.label} {self.sent_status}{retried}")
                    return
                if self.scanner.data:
                    log(f"{self.command} {self.label}: aborted after data, closing")
                    self.abandon("upstream changed after the response started")
                    return
                if upstream.aborted:
                    self.retries += 1
                    log(f"{self.command} {self.label}: aborted, retrying")
                else:
                    if time.monotonic() - opened > BACKOFF_CAP_SECONDS:
                        # The response was up a while: this is a new outage.
                        self.failing_since = None
                    try:
                        self.unreachable(relay.config, "the response broke")
                    except UpstreamError as error:
                        self.fail(str(error))
                        return
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
            elif lower == RETRY_COUNT_HEADER and self.sent:
                continue
            headers[name] = f"{headers[name]}, {value}" if name in headers else value
        if self.sent:
            headers[RETRY_COUNT_HEADER] = str(self.retry_count + self.sent)
        return headers

    def open(
        self,
    ) -> tuple[Upstream, http.client.HTTPConnection, http.client.HTTPResponse]:
        relay = self.server.relay
        while True:
            upstream, config = relay.begin()
            connection = config.connection()
            try:
                connection.connect()
                assert connection.sock is not None
                upstream.attach(connection.sock)
                # Otherwise only the connect is bounded: a model call may wait long for
                # its first byte, and the client's own timeout covers a dead upstream.
                connection.sock.settimeout(
                    SILENT_UPSTREAM_SECONDS if self.keeps_client_alive else None
                )
                connection.request(
                    self.command,
                    config.prefix + self.target,
                    self.body or None,
                    self.upstream_headers(config),
                )
                self.sent += 1
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as error:
                relay.end(upstream)
                connection.close()
                if upstream.aborted:
                    self.retries += 1
                    log(f"{self.command} {self.label}: aborted, retrying")
                    continue
                self.unreachable(config, f"{error!r}")
                continue
            if is_edge_error(response):
                # Not read: an edge that stalls its page must not block the retry.
                self.sent -= 1
                relay.end(upstream)
                connection.close()
                self.unreachable(config, f"tunnel edge answered {response.status}")
                continue
            return upstream, connection, response

    def unreachable(self, config: Config, reason: str) -> None:
        """Wait before the next attempt while the upstream cannot be reached: until the
        config changes, at most `RETRY_SECONDS` after the first failure."""
        now = time.monotonic()
        if self.failing_since is None:
            self.failing_since = now
            self.delay = BACKOFF_SECONDS
        if now - self.failing_since >= RETRY_SECONDS:
            raise UpstreamError(
                f"upstream unreachable for {RETRY_SECONDS:g}s: {reason}"
            )
        log(f"{config.origin} unreachable ({reason}), retry in {self.delay}s")
        self.retries += 1
        self.server.relay.wait(config, self.delay)
        self.delay = min(self.delay * 2, BACKOFF_CAP_SECONDS)

    @property
    def keeps_client_alive(self) -> bool:
        """Whether the relay sends the client keepalives while it waits for a retry: an
        SSE response it already started."""
        return self.sent_status is not None and self.scanner.sse

    @contextlib.contextmanager
    def keepalive(self) -> Iterator[None]:
        """Keep an SSE client's idle timer from firing while its request waits for a
        retry."""
        if not self.keeps_client_alive:
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
            self.fail(STATUS_CHANGED)
            return True
        elif self.sent_length is not None and response.length != self.sent_length:
            self.abandon("retry answered a body of another length")
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
                if not self.scanner.data:
                    # No body data went out: the retry can still answer.
                    return False
                self.abandon(f"upstream broke mid-body: {error!r}")
                return True
            # Data read after the abort is dropped: the retry sends it again.
            if upstream.aborted:
                return False
            if not data:
                if response.length:
                    # The upstream ended before its `Content-Length`.
                    log(f"{self.command} {self.label}: upstream body cut short")
                    if not self.scanner.data:
                        return False
                    self.abandon("upstream body cut short")
                    return True
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
        elif (
            response.length is not None
            and not response.chunked
            and not self.scanner.sse
        ):
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
        self.gave_up = True
        self.server.relay.record_failure(message)
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

    def abandon(self, message: str) -> None:
        """Give up on a response the client already has part of."""
        self.gave_up = True
        self.server.relay.record_failure(message)
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
    parser.add_argument("--failures")
    args = parser.parse_args()
    relay = Relay(args.config, args.failures)
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
