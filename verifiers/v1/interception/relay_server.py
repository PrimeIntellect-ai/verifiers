"""Loopback relay to the host's interception URL, run inside a remote runtime (stdlib only).

python3 relay_server.py UPSTREAM FILES WINDOW_SECONDS RETRY_HEADER STAMP_HEADER

Each request goes to UPSTREAM on a fresh connection and its response streams back. The
host stamps its every response with STAMP_HEADER, so a transient-looking error without it
came from something in between (a tunnel, proxy, or edge). Those, and failed connections,
are retried for up to WINDOW_SECONDS. A request that may have reached the host is only
repeated on routes the host dedupes, marked with RETRY_HEADER. FILES is a path prefix: the
relay writes FILES.port and FILES.json (counters) and reads proxy settings from FILES.env
whenever the host rewrites it.
"""

from __future__ import annotations

import base64
import email.errors
import hashlib
import http.client
import json
import math
import os
import random
import re
import select
import socket
import ssl
import sys
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

HOP_BY_HOP = {
    "connection",
    "expect",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
TARGET = re.compile(r"/[!-~]*")  # origin-form, visible ASCII
LENGTH = re.compile(r"[0-9]{1,15}")
MAX_BODY = 1 << 30  # the host's own limit
CHUNK = re.compile(rb"([0-9a-fA-F]{1,15})(;[^\r\n]*)?\r?\n")
FOLD = re.compile(r"\r?\n[ \t]+")
# What the header parser reports for a malformed header block (not, say, a multipart
# body it can't see).
MALFORMED = (
    email.errors.MissingHeaderBodySeparatorDefect,
    email.errors.FirstHeaderLineIsContinuationDefect,
    email.errors.MisplacedEnvelopeHeaderDefect,
    email.errors.InvalidHeaderDefect,
)


class Gone(Exception):
    """The client hung up."""


class Unrecoverable(Exception):
    """A failure retrying can't fix: a bad certificate, or a proxy refusing the host."""


def transient(status: int) -> bool:
    return status in (404, 408, 429) or status >= 500


def unrecoverable(error: Exception) -> bool:
    """A bad certificate, or a proxy refusing the host for good: retrying can't help."""
    refused = re.match(r"Tunnel connection failed: (\d+)", str(error))
    return isinstance(error, ssl.SSLCertVerificationError) or bool(
        refused and not transient(int(refused.group(1)))
    )


def final(response: http.client.HTTPResponse, method: str) -> http.client.HTTPResponse:
    """Skip interim responses (say, an edge's 103 Early Hints); the final one follows."""
    while 102 <= response.status < 200:
        fp, response.fp = response.fp, None
        reader = types.SimpleNamespace(makefile=lambda *args, fp=fp: fp)
        response = http.client.HTTPResponse(reader, method=method)
        response.begin()
    return response


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    server: RelayServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def read_body(self) -> bytes | None:
        """The request body, or None if its framing is invalid or the client left mid-body."""
        lengths = self.headers.get_all("Content-Length") or []
        codings = self.headers.get_all("Transfer-Encoding")
        if not codings:
            if len(set(lengths)) > 1 or not all(LENGTH.fullmatch(n) for n in lengths):
                return None
            size = int(lengths[0]) if lengths else 0
            body = self.rfile.read(size) if size <= MAX_BODY else b""
            return body if len(body) == size else None
        if lengths or ",".join(codings).replace(" ", "").lower() != "chunked":
            return None
        chunks, total = [], 0
        while True:
            line = CHUNK.fullmatch(self.rfile.readline(1 << 16))
            if line is None:
                return None
            size = int(line.group(1), 16)
            if not size:
                break
            total += size
            if total > MAX_BODY:
                return None
            chunks.append(self.rfile.read(size))
            if len(chunks[-1]) != size or self.rfile.readline(3) not in (
                b"\r\n",
                b"\n",
            ):
                return None
        while True:  # trailers, dropped
            line = self.rfile.readline(1 << 16)
            if line in (b"\r\n", b"\n"):
                return b"".join(chunks)
            if not line.endswith(b"\n"):
                return None

    def repeatable(self) -> bool:
        # Safe to repeat: the host answers a marked repeat of a model call or tool-gate
        # check with the original result; the rest are reads and whole-state replaces.
        path = self.path.partition("?")[0]
        return (
            self.command in ("GET", "HEAD")
            or path.startswith("/v1/")
            or path in ("/tool", "/state")
        )

    def hold(self, seconds: float) -> None:
        """Wait between attempts; raise Gone if the client hangs up meanwhile."""
        poll = select.poll()
        poll.register(self.connection, getattr(select, "POLLRDHUP", 0))
        if poll.poll(seconds * 1000):  # hang-ups and errors only, not new requests
            raise Gone

    def attempt(self, headers: list, body: bytes):
        """(connection, response, sent). On failure the first two are None, and `sent`
        says whether the request may have reached the host."""
        conn, sent = None, False
        try:
            conn = self.server.connect()
            self.hold(0)  # it may have given up while this connected
            sent = True
            response = self.send(conn, headers, body)
            if response.status < 200:  # 101: nothing here asked to switch protocols
                raise http.client.HTTPException(f"unexpected {response.status}")
            return conn, response, sent
        except Gone:
            conn.close()
            raise
        except (OSError, http.client.HTTPException) as e:
            if conn is not None:
                conn.close()
            if unrecoverable(e):
                raise Unrecoverable(e) from e
            return None, None, sent

    def send(self, conn, headers: list, body: bytes) -> http.client.HTTPResponse:
        conn.putrequest(
            self.command,
            self.server.base_path + self.path,
            skip_host=True,
            skip_accept_encoding=True,
        )
        for name, value in headers:
            conn.putheader(name, value)
        if body or self.command in ("POST", "PUT", "PATCH"):
            conn.putheader("Content-Length", str(len(body)))
        try:
            conn.endheaders(body or None)
        except OSError:
            pass  # the host may have answered (say, a 413) before taking it all
        return final(conn.getresponse(), self.command)

    def relay(self) -> None:
        server = self.server
        if self.request_version != "HTTP/1.1":
            self.close_connection = True
        body = self.read_body()
        connection = {
            token.strip().lower()
            for token in ",".join(self.headers.get_all("Connection") or []).split(",")
        }
        if "close" in connection:
            self.close_connection = True
        headers = [
            (name, FOLD.sub(" ", value))
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP | connection
        ]
        if (
            body is None
            or not TARGET.fullmatch(self.path)
            # A malformed line ends the headers early.
            or any(isinstance(d, MALFORMED) for d in self.headers.defects)
            or any("\r" in value or "\n" in value for _, value in headers)
        ):
            self.close_connection = True
            return self.reply(400, b"relay: malformed request")
        headers.append(("Host", server.host))
        repeatable = self.repeatable()
        held, delay, retry, arrived = 0.0, 0.5, 0, time.monotonic()
        key = hashlib.sha256(f"{self.command} {self.path} ".encode() + body).digest()
        # Its identical copy outlasted the window shortly before this one arrived: the
        # harness retrying it.
        retried = server.gave_up.get(key, -math.inf) > arrived - server.window
        try:
            while True:
                if retry and repeatable:
                    name = server.retry_header
                    headers = [(k, v) for k, v in headers if k.lower() != name.lower()]
                    headers.append((name, str(retry)))
                try:
                    conn, response, sent = self.attempt(headers, body)
                except Unrecoverable as e:
                    print(f"relay: {e!r}", file=sys.stderr, flush=True)
                    server.count(retry, held, rescued=False)
                    self.close_connection = True  # as a direct connection would fail
                    return
                if response is not None and (
                    response.getheader(server.stamp) or not transient(response.status)
                ):
                    break
                held = held or time.monotonic()
                left = held + server.window - time.monotonic()
                # The harness retrying a request that just outlasted the window fails
                # fast, so a longer outage ends the rollout as an error, not a timeout.
                if left <= 0:
                    server.give_up(key)
                elif not retried and (repeatable or not (response is not None or sent)):
                    if response is not None:
                        conn.close()
                    self.hold(min(delay * random.uniform(0.5, 1.5), left))
                    delay, retry = min(delay * 2, 10.0), retry + 1
                    continue
                server.count(retry, held, rescued=False)
                if response is not None:
                    return self.forward(conn, response)
                # As a direct connection would have seen it: no answer at all.
                self.close_connection = True
                return
        except Gone:
            server.count(retry, held, rescued=False)
            self.close_connection = True
            return
        if key in server.gave_up:
            server.give_up(key, recovered=True)
        server.count(retry, held, rescued=True)
        self.forward(conn, response)

    def reply(self, status: int, payload: bytes) -> None:
        self.send_response_only(status)
        self.send_header("Content-Length", str(len(payload)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def forward(self, conn, response: http.client.HTTPResponse) -> None:
        no_body = self.command == "HEAD" or response.status in (204, 304)
        # Framed as http.client reads the body, not as the upstream's headers claim.
        length = response.length
        if no_body:
            length = response.getheader("Content-Length", "")
            length = int(length) if LENGTH.fullmatch(length) else None
        chunked = not no_body and length is None and self.request_version == "HTTP/1.1"
        if not no_body and not chunked and length is None:
            self.close_connection = True  # the body ends when the connection does
        self.send_response_only(response.status, response.reason)
        for name, value in response.getheaders():
            if name.lower() not in HOP_BY_HOP:
                self.send_header(name, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        elif length is not None:
            self.send_header("Content-Length", str(length))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        complete = False
        try:
            while not no_body:
                data = response.read1(1 << 16)
                if not data:
                    break
                self.wfile.write(
                    b"%x\r\n%s\r\n" % (len(data), data) if chunked else data
                )
                self.wfile.flush()
            # A body the host's side cut short must not look complete to the client.
            if no_body or not response.length:
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
                complete = True
        finally:
            response.close()
            conn.close()
            if not complete:
                self.close_connection = True

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = relay


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024

    def __init__(
        self,
        port: int,
        upstream: str,
        files: str,
        window: float,
        retry_header: str,
        stamp: str,
    ):
        super().__init__(("127.0.0.1", port), Relay)
        url = urlsplit(upstream)
        self.https = url.scheme == "https"
        self.hostname, self.host = url.hostname, url.netloc.rpartition("@")[2]
        self.port = url.port or (443 if self.https else 80)
        self.base_path = url.path.rstrip("/")
        self.context = ssl.create_default_context() if self.https else None
        self.files, self.window, self.retry_header = files, window, retry_header
        self.stamp = stamp
        self.env_mtime = 0.0
        self.gave_up: dict = {}
        self.lock = threading.Lock()
        self.stats = {
            "relay_retried_requests": 0,
            "relay_rescued_requests": 0,
            "relay_retry_seconds": 0.0,
        }
        try:
            with open(f"{files}.json") as f:
                saved = json.load(f)
            for key in self.stats:
                if isinstance(saved.get(key), (int, float)):
                    self.stats[key] = saved[key]
        except (OSError, ValueError, AttributeError):
            pass
        self.save()

    def load_env(self) -> None:
        """Adopt the proxy settings the runtime gives programs once its network policy
        applies (written by the host after the relay started)."""
        try:
            mtime = os.stat(f"{self.files}.env").st_mtime
            if mtime == self.env_mtime:
                return
            with open(f"{self.files}.env") as f:
                lines = [line.rstrip("\n").split("=", 1) for line in f if "=" in line]
            env = {k: v for k, v in lines if k.lower().endswith("_proxy")}
        except OSError:
            return
        with self.lock:
            for key in [k for k in os.environ if k.lower().endswith("_proxy")]:
                os.environ.pop(key)
            os.environ.update(env)
            self.env_mtime = mtime

    def connect(self) -> http.client.HTTPConnection:
        self.load_env()
        host, port, proxy = self.hostname, self.port, None
        with self.lock:
            if not urllib.request.proxy_bypass_environment(host):
                proxy = urllib.request.getproxies_environment().get(
                    "https" if self.https else "http"
                )
        if proxy:
            proxy = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
            host, port = proxy.hostname, proxy.port or 80
        if self.https:
            conn = http.client.HTTPSConnection(
                host, port, timeout=30, context=self.context
            )
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)
        if proxy:
            headers = {}
            if proxy.username:
                auth = f"{unquote(proxy.username)}:{unquote(proxy.password or '')}"
                headers["Proxy-Authorization"] = (
                    f"Basic {base64.b64encode(auth.encode()).decode()}"
                )
            conn.set_tunnel(self.hostname, self.port, headers)
        conn.connect()
        conn.sock.settimeout(None)
        # Model turns can stream for many minutes, so no read timeout. Keepalive probes,
        # and a cap on unacknowledged sends, notice a connection that died silently.
        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for option, value in (
            ("TCP_KEEPIDLE", 30),
            ("TCP_KEEPINTVL", 10),
            ("TCP_KEEPCNT", 6),
            ("TCP_USER_TIMEOUT", 90_000),
        ):
            if hasattr(socket, option):
                conn.sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)
        return conn

    def check(self) -> str:
        """Why requests through here can't reach the host: "" if they can, or if the
        network is just down (retries handle that)."""
        conn = None
        try:
            conn = self.connect()
            conn.sock.settimeout(30)
            headers = {"Host": self.host}
            conn.request("GET", f"{self.base_path}/v1/models", headers=headers)
            response = final(conn.getresponse(), "GET")
            if response.getheader(self.stamp) or transient(response.status):
                return ""
            return f"its address answered {response.status} without the host's stamp"
        except ssl.SSLError as e:
            # Any TLS failure but a dropped connection: this relay won't get through.
            dropped = isinstance(e, (ssl.SSLEOFError, ssl.SSLZeroReturnError))
            return "" if dropped else repr(e)
        except (OSError, http.client.HTTPException) as e:
            return repr(e) if unrecoverable(e) else ""
        except Exception as e:  # noqa: BLE001 - a bad setting, not the network
            return repr(e)
        finally:
            if conn is not None:
                conn.close()

    def give_up(self, key: bytes, recovered: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            self.gave_up = {
                k: t for k, t in self.gave_up.items() if t > now - self.window
            }
            if recovered:
                self.gave_up.pop(key, None)
            else:
                self.gave_up[key] = now

    def handle_error(self, request, client_address) -> None:
        print(f"relay: {sys.exc_info()[1]!r}", file=sys.stderr, flush=True)

    def count(self, retry: int, held: float, rescued: bool) -> None:
        if retry:
            with self.lock:
                self.stats["relay_retried_requests"] += 1
                self.stats["relay_rescued_requests"] += rescued
                self.stats["relay_retry_seconds"] += time.monotonic() - held
                self.save()

    def save(self) -> None:
        try:
            with open(f"{self.files}.json.tmp", "w") as f:
                json.dump(self.stats, f)
            os.replace(f"{self.files}.json.tmp", f"{self.files}.json")
        except OSError:
            pass


def main() -> None:
    upstream, files, window, retry_header, stamp = sys.argv[1:6]
    try:
        with open(f"{files}.port") as f:
            port = int(f.read())  # a restart keeps the address
    except (OSError, ValueError):
        port = 0
    try:
        server = RelayServer(port, upstream, files, float(window), retry_header, stamp)
        # Check the way to the host on first start; a restart serves at once, while
        # connections wait in the listen queue.
        refused = "" if port else server.check()
    except Exception as e:
        if port:
            raise  # the loop tries again
        refused = f"crashed: {e!r}"
    with open(f"{files}.port.tmp", "w") as f:
        f.write(
            f"refused: {refused[:400]}" if refused else str(server.server_address[1])
        )
    os.replace(f"{files}.port.tmp", f"{files}.port")
    if refused:
        threading.Event().wait()  # the host goes direct and stops this
    server.serve_forever()


if __name__ == "__main__":
    main()
