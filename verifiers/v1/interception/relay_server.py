"""Loopback relay to a tunneled host URL, run inside a remote runtime (stdlib only).

python3 relay_server.py UPSTREAM FILES WINDOW_SECONDS RETRY_HEADER STAMP_HEADER

The host stamps its every response with STAMP_HEADER, so a transient-looking error without
it came from something in between (a tunnel, proxy, or edge) and is retried, whatever its
wording. FILES is a path prefix: the relay writes FILES.port and FILES.json (counters) and
reads proxy settings from FILES.env whenever the host rewrites it.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import random
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


def marked(headers: list, name: str, retry: int) -> list:
    kept = [(k, v) for k, v in headers if k.lower() != name.lower()]
    return [*kept, (name, str(retry))]


def transient(status: int) -> bool:
    return status in (404, 408, 429) or status >= 500


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RelayServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def read_body(self) -> bytes:
        if "chunked" not in (self.headers.get("Transfer-Encoding") or "").lower():
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))
        chunks = []
        while size := int(self.rfile.readline().split(b";")[0], 16):
            chunks.append(self.rfile.read(size))
            self.rfile.readline()
        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
            pass
        return b"".join(chunks)

    def repeatable(self) -> bool:
        # Safe to repeat: the host answers a marked repeat of a model call or tool-gate
        # check with the original result; the rest are reads and whole-state replaces.
        path = urlsplit(self.path).path
        return (
            self.command in ("GET", "HEAD")
            or path.startswith("/v1/")
            or path in ("/tool", "/state")
        )

    def attempt(self, headers: list, body: bytes, repeatable: bool):
        """(connection, response, sent); on failure the first two are None and `sent`
        says whether the request may have reached the host."""
        server = self.server
        with server.lock:
            # Only repeatable requests reuse connections: a stale one fails after
            # sending, and is then retried at once on a fresh connection.
            conn = server.idle.pop() if repeatable and server.idle else None
        if conn is not None:
            try:
                return conn, self.send(conn, headers, body), True
            except (OSError, http.client.HTTPException):
                conn.close()
                headers = marked(headers, server.retry_header, 1)
        conn, sent = None, False
        try:
            conn = server.connect()
            sent = True
            return conn, self.send(conn, headers, body), sent
        except ssl.SSLCertVerificationError:
            raise
        except (OSError, http.client.HTTPException):
            if conn is not None:
                conn.close()
            return None, None, sent

    def send(self, conn, headers: list, body: bytes) -> http.client.HTTPResponse:
        conn.putrequest(
            self.command, self.server.base_path + self.path, skip_accept_encoding=True
        )
        for name, value in headers:
            conn.putheader(name, value)
        if body or self.command in ("POST", "PUT", "PATCH"):
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body or None)
        return conn.getresponse()

    def relay(self) -> None:
        body = self.read_body()
        headers = [
            (k, v) for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP
        ]
        headers.append(("Host", self.server.host))
        repeatable = self.repeatable()
        failed_at, delay, retry = 0.0, 0.5, 0
        while True:
            if retry and repeatable:
                headers = marked(headers, self.server.retry_header, retry)
            try:
                conn, response, sent = self.attempt(headers, body, repeatable)
            except ssl.SSLCertVerificationError as e:
                return self.reply(502, f"relay: {e}".encode())
            page = b""
            if response is not None:
                if response.getheader(self.server.stamp) or not transient(
                    response.status
                ):
                    break
                # An error from something between here and the host, which may or may
                # not have passed the request on.
                page = response.read(1 << 16)
                conn.close()
                if not repeatable:
                    return self.reply(response.status, page, response.getheaders())
            elif sent and not repeatable:
                return self.reply(502, b"relay: connection to host lost")
            failed_at = failed_at or time.monotonic()
            pause = delay * random.uniform(0.5, 1.5)
            if time.monotonic() + pause > failed_at + self.server.window:
                self.server.count(retry, failed_at, rescued=False)
                if response is not None:
                    return self.reply(response.status, page, response.getheaders())
                return self.reply(502, b"relay: host unreachable")
            time.sleep(pause)
            delay, retry = min(delay * 2, 10.0), retry + 1
        self.server.count(retry, failed_at, rescued=True)
        self.forward(conn, response, page)

    def reply(self, status: int, payload: bytes, headers: list = ()) -> None:
        self.send_response_only(status)
        for name, value in headers:
            if name.lower() not in HOP_BY_HOP:
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def forward(self, conn, response: http.client.HTTPResponse, data: bytes) -> None:
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
        elif length is not None:
            self.send_header("Content-Length", length)
        self.end_headers()
        clean = False
        try:
            while not no_body:
                if data:
                    self.wfile.write(
                        b"%x\r\n%s\r\n" % (len(data), data) if chunked else data
                    )
                    self.wfile.flush()
                data = response.read1(1 << 16)
                if not data:
                    if chunked:
                        self.wfile.write(b"0\r\n\r\n")
                    break
            clean = True
        finally:
            if clean and not response.will_close:
                with self.server.lock:
                    self.server.idle.append(conn)
            else:
                conn.close()
                self.close_connection = True

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = relay


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, upstream: str, files: str, window: float, retry_header: str, stamp: str
    ):
        try:
            with open(f"{files}.port") as f:
                port = int(f.read().split()[0])  # a restart keeps the address
        except (OSError, ValueError, IndexError):
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
        self.idle: list = []
        self.lock = threading.Lock()
        self.stats = {
            "relay_retried_requests": 0,
            "relay_rescued_requests": 0,
            "relay_retry_seconds": 0.0,
        }
        try:
            with open(f"{files}.json") as f:
                self.stats.update(json.load(f))
        except (OSError, ValueError):
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
        # Model turns can stream for many minutes, so no read timeout; keepalive probes
        # instead notice a connection that died silently (about 90 s).
        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for option, value in (
            ("TCP_KEEPIDLE", 30),
            ("TCP_KEEPINTVL", 10),
            ("TCP_KEEPCNT", 6),
        ):
            if hasattr(socket, option):
                conn.sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)
        return conn

    def handle_error(self, request, client_address) -> None:
        print(f"relay: {sys.exc_info()[1]!r}", file=sys.stderr, flush=True)

    def count(self, retry: int, started: float, rescued: bool) -> None:
        if retry:
            with self.lock:
                self.stats["relay_retried_requests"] += 1
                self.stats["relay_rescued_requests"] += rescued
                self.stats["relay_retry_seconds"] += time.monotonic() - started
                self.save()

    def save(self) -> None:
        with open(f"{self.files}.json.tmp", "w") as f:
            json.dump(self.stats, f)
        os.replace(f"{self.files}.json.tmp", f"{self.files}.json")


def main() -> None:
    upstream, files, window, retry_header, stamp = sys.argv[1:6]
    server = RelayServer(upstream, files, float(window), retry_header, stamp)
    with open(f"{files}.port.tmp", "w") as f:
        f.write(f"{server.server_address[1]} {os.getpid()}")
    os.replace(f"{files}.port.tmp", f"{files}.port")
    server.serve_forever()


if __name__ == "__main__":
    main()
