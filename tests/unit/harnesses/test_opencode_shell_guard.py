"""OpenCode's shell guard: a command holding a NUL byte, or one printing without end,
must end its tool call instead of wedging OpenCode's session (the turn then never ends).
The binary test runs against a scripted local chat server and needs `VF_OPENCODE_BIN`
(an `opencode` 2.x executable) and `node`."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from verifiers.v1.harnesses.opencode.harness import (
    SHELL_CAP,
    OpenCodeHarness,
    OpenCodeHarnessConfig,
)
from verifiers.v1.task import TaskData

NODE = shutil.which("node")


class Runtime:
    """Writes the harness's files on this machine, at the paths the box would hold."""

    async def write(self, path: str, data: bytes) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


async def opencode_env(endpoint: str, trace_id: str, **config) -> dict[str, str]:
    harness = OpenCodeHarness(OpenCodeHarnessConfig(**config))
    acp = await harness.prepare_acp(
        SimpleNamespace(model="m"),
        SimpleNamespace(id=trace_id),
        Runtime(),
        endpoint,
        "k",
        {},
        TaskData(prompt="go"),
    )
    return acp.env


def guard(settings: dict) -> dict:
    (plugin,) = [
        p for p in settings["plugins"] if p["package"].endswith("/shell-guard")
    ]
    return plugin


def test_guard_plugin_and_its_cap_reach_the_box():
    trace_id = f"guard-{uuid.uuid4().hex}"
    root = Path("/tmp/vf-opencode-runs") / trace_id
    try:
        env = asyncio.run(opencode_env("http://e/v1", trace_id))
        plugin = guard(json.loads(env["OPENCODE_CONFIG_CONTENT"]))
        assert plugin["options"] == {
            "maxOutputBytes": 64 * 1024 * 1024,
            "node": "/var/tmp/vf-node/bin/node",
            "cap": f"{root}/shell-guard/cap.js",
        }
        assert (root / "shell-guard" / "index.js").is_file()
        assert (root / "shell-guard" / "cap.js").read_bytes() == SHELL_CAP
        env = asyncio.run(
            opencode_env("http://e/v1", trace_id, max_shell_output_bytes=None)
        )
        # No cap: the guard still turns away a NUL byte.
        assert (
            guard(json.loads(env["OPENCODE_CONFIG_CONTENT"]))["options"][
                "maxOutputBytes"
            ]
            is None
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_cap_passes_the_head_and_drops_the_rest():
    with tempfile.TemporaryDirectory() as root:
        cap = Path(root) / "cap.js"
        cap.write_bytes(SHELL_CAP)

        def run(data: bytes, limit: int) -> bytes:
            return subprocess.run(
                [NODE, str(cap), str(limit)],
                input=data,
                capture_output=True,
                check=True,
            ).stdout

        assert run(b"abc\0def", 100) == b"abc\0def"
        assert run(b"abcdef", 6) == b"abcdef"
        assert run(b"abcdefgh", 4) == b"abcd\n[output past 4 bytes dropped]\n"
        big = run(b"x" * 5_000_000, 1_000_000)
        assert big == b"x" * 1_000_000 + b"\n[output past 1000000 bytes dropped]\n"


COMMANDS = [
    # A NUL byte in argv: OpenCode 2.1.4 left this tool call running forever.
    "printf 'a\0b'",
    # More output than the cap.
    "head -c 3000000 /dev/zero | tr '\\0' x; echo; echo tail",
    # A background job holding the output open past the shell's exit, and an exit code.
    "echo out; echo err >&2; (sleep 5; echo late) & echo bg; exit 7",
]


class Scripted(BaseHTTPRequestHandler):
    """Turn n runs COMMANDS[n-1] in the shell; then the model says it is done."""

    requests: ClassVar[list[dict]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(body)
        n = len(type(self).requests)
        if n > len(COMMANDS):
            deltas, finish = [{"role": "assistant", "content": "Done."}], "stop"
        else:
            args = json.dumps({"command": COMMANDS[n - 1], "description": "x"})
            call = {"index": 0, "id": f"call_{n}", "type": "function"}
            deltas = [
                {"role": "assistant", "content": f"Turn {n}."},
                {
                    "tool_calls": [
                        {**call, "function": {"name": "shell", "arguments": args}}
                    ]
                },
            ]
            finish = "tool_calls"
        base = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
        }
        events = [
            *({**base, "choices": [{"index": 0, "delta": d}]} for d in deltas),
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.mark.skipif(
    not os.environ.get("VF_OPENCODE_BIN") or NODE is None,
    reason="needs VF_OPENCODE_BIN (opencode 2.x) and node",
)
def test_opencode_ends_nul_and_oversized_shell_calls():
    Scripted.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
    trace_id = f"guard-{uuid.uuid4().hex}"
    try:
        with tempfile.TemporaryDirectory() as root:
            env = asyncio.run(
                opencode_env(endpoint, trace_id, max_shell_output_bytes=1_000_000)
            )
            settings = json.loads(env["OPENCODE_CONFIG_CONTENT"])
            guard(settings)["options"]["node"] = NODE  # the box's node is not here
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(settings)
            for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
                env[key] = f"{root}/{key}"
            env["XDG_STATE_HOME"] = f"{root}/state"
            env["OPENCODE_CONFIG_DIR"] = f"{root}/config"
            work = Path(root) / "work"
            work.mkdir()
            subprocess.run(
                [os.environ["VF_OPENCODE_BIN"], "run", "--standalone", "--auto", "go"],
                cwd=work,
                env={"PATH": os.environ["PATH"], "HOME": root, **env},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=120,
                check=True,
            )
            outputs = list(Path(root, "XDG_DATA_HOME").rglob("sh_*.out"))
            assert outputs and max(p.stat().st_size for p in outputs) < 1_100_000
    finally:
        server.shutdown()
        shutil.rmtree(Path("/tmp/vf-opencode-runs") / trace_id, ignore_errors=True)

    results = [
        str([m for m in r["messages"] if m["role"] == "tool"][-1]["content"])
        for r in Scripted.requests[1:]
    ]
    assert len(results) == len(COMMANDS)
    nul, oversized, background = results
    assert "contains a NUL byte" in nul and "code 2" in nul
    # OpenCode shows the output's tail (then where it keeps the whole of it).
    assert "x\n[output past 1000000 bytes dropped]\n" in oversized
    assert "\ntail\n" not in oversized
    for line in ("out", "err", "bg"):
        assert line in background.splitlines()
    assert "code 7" in background and "late" not in background
