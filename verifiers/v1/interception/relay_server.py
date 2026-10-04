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
import http.client
import json
import os
import random
import re
import select
import socket
import ssl
import sys
import threading
import time
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
CHUNK = re.compile(rb"([0-9a-fA-F]{1,15})(;[^\r\n]*)?\r?\n")
FOLD = re.compile(r"\r?\n[ \t]+")


class Gone(Exception):
    """The client hung up."""


class Unrecoverable(Exception):
    """A failure retrying can't fix: a bad certificate, or a proxy refusing the host."""


def transient(status: int) -> bool:
    return status in (404, 408, 429) or status >= 500


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RelayServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def read_body(self) -> bytes | None:
        """The request body, or None if its framing is invalid or the client left mid-body."""
        lengths = self.headers.get_all("Content-Length") or []
        coding = self.headers.get("Transfer-Encoding")
        if coding is None:
            if len(set(lengths)) > 1 or not all(LENGTH.fullmatch(n) for n in lengths):
                return None
            size = int(lengths[0]) if lengths else 0
            body = self.rfile.read(size)
            return body if len(body) == size else None
        if lengths or coding.strip().lower() != "chunked":
            return None
        chunks = []
        while True:
            line = CHUNK.fullmatch(self.rfile.readline(1 << 16))
            if line is None:
                return None
            size = int(line.group(1), 16)
            if not size:
                break
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
        deadline = time.monotonic() + seconds
        try:
            ready = select.select([self.connection], [], [], seconds)[0]
            if ready and not self.connection.recv(1, socket.MSG_PEEK):
                raise Gone
        except OSError:
            raise Gone from None
        # Readable but not closed: the client sent its next request early.
        time.sleep(max(0.0, deadline - time.monotonic()))

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
        except ssl.SSLCertVerificationError as e:
            if conn is not None:
                conn.close()
            raise Unrecoverable(e) from e
        except (OSError, http.client.HTTPException) as e:
            if conn is not None:
                conn.close()
            refused = re.match(r"Tunnel connection failed: (\d+)", str(e))
            if refused and not transient(int(refused.group(1))):
                raise Unrecoverable(e) from e
            return None, None, sent
        except Gone:
            if conn is not None:
                conn.close()
            raise

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
        response = conn.getresponse()
        # Skip interim responses (say, an edge's 103 Early Hints); the final one follows.
        while 102 <= response.status < 200:
            final = http.client.HTTPResponse(conn.sock, method=self.command)
            final.fp.close()
            final.fp, response.fp = response.fp, None
            final.begin()
            response = final
        return response

    def relay(self) -> None:
        server = self.server
        if self.request_version != "HTTP/1.1":
            self.close_connection = True
        body = self.read_body()
        connection = {
            token.strip().lower()
            for token in (self.headers.get("Connection") or "").split(",")
        }
        headers = [
            (name, FOLD.sub(" ", value))
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP | connection
        ]
        if (
            body is None
            or not TARGET.fullmatch(self.path)
            or any("\r" in value or "\n" in value for _, value in headers)
        ):
            self.close_connection = True
            return self.reply(400, b"relay: malformed request")
        headers.append(("Host", server.host))
        repeatable = self.repeatable()
        held, delay, retry = 0.0, 0.5, 0
        try:
            while True:
                if retry and repeatable:
                    name = server.retry_header
                    headers = [(k, v) for k, v in headers if k.lower() != name.lower()]
                    headers.append((name, str(retry)))
                try:
                    conn, response, sent = self.attempt(headers, body)
                except Unrecoverable as e:
                    return self.reply(502, f"relay: {e}".encode())
                if response is not None and (
                    response.getheader(server.stamp) or not transient(response.status)
                ):
                    break
                if response is None and sent and not repeatable:
                    return self.reply(502, b"relay: connection to host lost")
                held = held or time.monotonic()
                pause = delay * random.uniform(0.5, 1.5)
                if time.monotonic() + pause > held + server.window:
                    # Failures soon after fail fast too, so the harness's own retries
                    # don't each wait out the window: a longer outage ends the rollout
                    # as an error, not a timeout.
                    server.fail_fast_until = time.monotonic() + server.window
                if time.monotonic() < server.fail_fast_until or (
                    response is not None and not repeatable
                ):
                    server.count(retry, held, rescued=False)
                    if response is not None:
                        return self.forward(conn, response)
                    return self.reply(502, b"relay: host unreachable")
                if response is not None:
                    conn.close()
                self.hold(pause)
                delay, retry = min(delay * 2, 10.0), retry + 1
        except Gone:
            server.count(retry, held, rescued=False)
            self.close_connection = True
            return
        server.fail_fast_until = 0.0
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
        length = response.getheader("Content-Length")
        chunked = (
            not no_body
            and (response.chunked or length is None)
            and self.request_version != "HTTP/1.0"
        )
        if not no_body and not chunked and length is None:
            self.close_connection = True  # the body ends when the connection does
        self.send_response_only(response.status, response.reason)
        for name, value in response.getheaders():
            if name.lower() not in HOP_BY_HOP:
                self.send_header(name, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        elif length is not None and not response.chunked:
            self.send_header("Content-Length", length)
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
            conn.close()
            if not complete:
                self.close_connection = True

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = relay


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, upstream: str, files: str, window: float, retry_header: str, stamp: str
    ):
        try:
            with open(f"{files}.port") as f:
                port = int(f.read())  # a restart keeps the address
        except (OSError, ValueError):
            port = 0
        super().__init__(("127.0.0.1", port), Relay)
        url = urlsplit(upstream)
        self.https = url.scheme == "https"
        self.hostname, self.host = url.hostname, url.netloc
        self.port = url.port or (443 if self.https else 80)
        self.base_path = url.path.rstrip("/")
        self.context = ssl.create_default_context() if self.https else None
        self.files, self.window, self.retry_header = files, window, retry_header
        self.stamp = stamp
        self.env_mtime = 0.0
        self.fail_fast_until = 0.0
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
    server = RelayServer(upstream, files, float(window), retry_header, stamp)
    with open(f"{files}.port.tmp", "w") as f:
        f.write(str(server.server_address[1]))
    os.replace(f"{files}.port.tmp", f"{files}.port")
    server.serve_forever()


if __name__ == "__main__":
    main()
