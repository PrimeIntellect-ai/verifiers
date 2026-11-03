"""The in-box relay forwards requests to its upstream with swapped tokens, streams
responses as they arrive, and on SIGHUP moves requests still waiting for their first
event to the new upstream on the same client connection."""

import http.client
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from verifiers.v1.runtimes.relay import RELAY_SOURCE, relay_config

TIMEOUT = 5


class FakeUpstream:
    """An HTTP server whose handler each test sets; it records every request."""

    def __init__(self, port: int = 0) -> None:
        self.requests: queue.Queue[dict] = queue.Queue()
        self.respond: Callable[[BaseHTTPRequestHandler], None] = lambda h: None
        self.stop = threading.Event()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args) -> None:
                pass

            def handle_request(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                upstream.requests.put(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": self.rfile.read(length),
                    }
                )
                try:
                    upstream.respond(self)
                except OSError:
                    self.close_connection = True

            do_GET = do_POST = handle_request

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        # The relay resets connections whose response it abandons.
        self.server.handle_error = lambda request, address: None
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever, args=(0.05,), daemon=True
        ).start()

    def close(self) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()


def reply(handler, status: int, body: dict) -> None:
    data = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def start_sse(handler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()


def chunk(handler, data: bytes) -> None:
    handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
    handler.wfile.flush()


def end_sse(handler) -> None:
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


def keepalives(upstream: FakeUpstream, before: bytes = b""):
    """An SSE response that sends `before`, then only keepalives until it is cut."""

    def respond(handler) -> None:
        start_sse(handler)
        if before:
            chunk(handler, before)
        while not upstream.stop.is_set():
            chunk(handler, b": keepalive\n")
            upstream.stop.wait(0.05)

    return respond


def events(*data: bytes):
    def respond(handler) -> None:
        start_sse(handler)
        for event in data:
            chunk(handler, event)
        end_sse(handler)

    return respond


class Relay:
    def __init__(
        self, tmp_path: Path, upstream: str, tokens: dict[str, str], source: str
    ) -> None:
        script = tmp_path / "relay_server.py"
        script.write_text(source)
        self.config = tmp_path / "config.json"
        self.write(upstream, tokens)
        ready = tmp_path / "port"
        self.stderr = (tmp_path / "relay.log").open("w+")
        self.process = subprocess.Popen(
            [sys.executable, str(script), "--config", str(self.config)]
            + ["--port", "0", "--ready", str(ready)],
            stderr=self.stderr,
        )
        self.port = int(wait_for(lambda: ready.exists() and ready.read_text()))

    def write(self, upstream: str, tokens: dict[str, str]) -> None:
        self.config.write_bytes(relay_config(upstream, tokens))

    def reload(self, upstream: str, tokens: dict[str, str]) -> None:
        self.write(upstream, tokens)
        os.kill(self.process.pid, signal.SIGHUP)

    def connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)

    def close(self) -> str:
        self.process.kill()
        self.process.wait()
        self.stderr.seek(0)
        return self.stderr.read()


def wait_for(condition: Callable, timeout: float = TIMEOUT):
    deadline = time.monotonic() + timeout
    while not (value := condition()):
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        time.sleep(0.01)
    return value


def read_until(response: http.client.HTTPResponse, marker: bytes) -> bytes:
    data = b""
    while marker not in data:
        piece = response.read1(65536)
        if not piece:
            raise EOFError(data)
        data += piece
    return data


@pytest.fixture
def upstreams() -> Iterator[tuple[FakeUpstream, FakeUpstream]]:
    a, b = FakeUpstream(), FakeUpstream()
    yield a, b
    a.close()
    b.close()


@pytest.fixture
def make_relay(tmp_path) -> Iterator[Callable[..., Relay]]:
    relays: list[Relay] = []

    def make(
        upstream: str, tokens: dict[str, str] | None = None, source: str = RELAY_SOURCE
    ) -> Relay:
        relays.append(Relay(tmp_path, upstream, tokens or {}, source))
        return relays[-1]

    yield make
    for relay in relays:
        sys.stderr.write(relay.close())


def sse_post(relay: Relay, token: str = "box") -> http.client.HTTPResponse:
    connection = relay.connect()
    connection.request(
        "POST",
        "/v1/chat/completions",
        b'{"stream": true}',
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    return connection.getresponse()


def test_forwards_with_swapped_tokens(upstreams, make_relay):
    upstream, _ = upstreams
    upstream.respond = lambda h: reply(h, 401, {"error": "nope"})
    relay = make_relay(f"{upstream.url}/base/", {"box": "real", "key": "real-key"})
    connection = relay.connect()
    for headers, expected in (
        ({"Authorization": "Bearer box"}, ("Bearer real", None)),
        ({"x-api-key": "key"}, (None, "real-key")),
        (
            {"Authorization": "Bearer other", "x-api-key": "other"},
            ("Bearer other", "other"),
        ),
    ):
        connection.request("POST", "/v1/messages?beta=true", b'{"a": 1}', headers)
        response = connection.getresponse()
        assert response.status == 401
        assert response.getheader("Content-Type") == "application/json"
        assert json.loads(response.read()) == {"error": "nope"}
        seen = upstream.requests.get(timeout=TIMEOUT)
        assert seen["method"] == "POST"
        assert seen["path"] == "/base/v1/messages?beta=true"
        assert seen["body"] == b'{"a": 1}'
        assert seen["headers"]["host"] == upstream.url.removeprefix("http://")
        auth = (seen["headers"].get("authorization"), seen["headers"].get("x-api-key"))
        assert auth == expected


def test_streams_sse_as_it_arrives(upstreams, make_relay):
    upstream, _ = upstreams
    received = threading.Event()

    def respond(handler) -> None:
        start_sse(handler)
        chunk(handler, b"data: 1\n\n")
        received.wait(TIMEOUT)
        chunk(handler, b"data: 2\n\n")
        end_sse(handler)

    upstream.respond = respond
    response = sse_post(make_relay(upstream.url))
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    assert read_until(response, b"data: 1\n\n") == b"data: 1\n\n"
    received.set()
    assert response.read() == b"data: 2\n\n"


def test_sighup_retries_waiting_request_on_new_upstream(upstreams, make_relay):
    a, b = upstreams
    a.respond = keepalives(a)
    b.respond = events(b"data: from-b\n\n", b"data: [DONE]\n\n")
    relay = make_relay(a.url, {"box": "secret-a"})
    response = sse_post(relay)
    assert response.status == 200
    read_until(response, b": keepalive\n")
    assert (
        a.requests.get(timeout=TIMEOUT)["headers"]["authorization"] == "Bearer secret-a"
    )
    relay.reload(b.url, {"box": "secret-b"})
    body = response.read()
    assert body.endswith(b"data: from-b\n\ndata: [DONE]\n\n")
    assert b"data:" not in body.partition(b"data: from-b")[0]
    seen = b.requests.get(timeout=TIMEOUT)
    assert seen["headers"]["authorization"] == "Bearer secret-b"
    assert seen["headers"]["x-stainless-retry-count"] == "1"
    assert seen["body"] == b'{"stream": true}'


def test_sighup_keeps_client_alive_until_new_upstream_is_up(upstreams, make_relay):
    a, _ = upstreams
    a.respond = keepalives(a)
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    source = RELAY_SOURCE.replace(
        "KEEPALIVE_SECONDS = 10.0", "KEEPALIVE_SECONDS = 0.05"
    )
    relay = make_relay(a.url, source=source)
    response = sse_post(relay)
    read_until(response, b": keepalive\n")
    relay.reload(f"http://127.0.0.1:{port}", {})
    read_until(response, b": relay keepalive\n")
    b = FakeUpstream(port)
    try:
        b.respond = events(b"data: from-b\n\n")
        assert response.read().endswith(b"data: from-b\n\n")
    finally:
        b.close()


def test_sighup_retry_with_other_status_sends_error_event(upstreams, make_relay):
    a, b = upstreams
    a.respond = keepalives(a)
    b.respond = lambda h: reply(h, 500, {"error": "boom"})
    relay = make_relay(a.url)
    response = sse_post(relay)
    read_until(response, b": keepalive\n")
    relay.reload(b.url, {})
    assert response.read().endswith(
        b'event: error\ndata: {"error": "relay: upstream changed status on retry"}\n\n'
    )


def test_sighup_after_data_closes_client_connection(upstreams, make_relay):
    a, b = upstreams
    a.respond = keepalives(a, before=b"data: partial\n\n")
    relay = make_relay(a.url)
    response = sse_post(relay)
    read_until(response, b"data: partial\n\n")
    relay.reload(b.url, {})
    with pytest.raises((http.client.HTTPException, OSError)):
        while response.read1(65536):
            pass
    assert b.requests.empty()


def test_concurrent_requests_do_not_block_each_other(upstreams, make_relay):
    upstream, _ = upstreams
    both = threading.Barrier(2, timeout=TIMEOUT)

    def respond(handler) -> None:
        both.wait()
        events(b"data: done\n\n")(handler)

    upstream.respond = respond
    relay = make_relay(upstream.url)
    bodies: list[bytes] = []

    def call() -> None:
        bodies.append(sse_post(relay).read())

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert bodies == [b"data: done\n\n"] * 2


def test_unreachable_upstream_answers_502(make_relay):
    relay = make_relay("http://127.0.0.1:1")
    started = time.monotonic()
    response = sse_post(relay)
    assert response.status == 502
    assert "connect failed" in json.loads(response.read())["error"]
    # A few short attempts, not the reload window.
    assert time.monotonic() - started < 5
