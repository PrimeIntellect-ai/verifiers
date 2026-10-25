"""OpenCode auto-compaction: the harness hands OpenCode the model's context window and
threshold, and a real OpenCode binary compacts there. The binary test runs against a
scripted local chat server and needs `VF_OPENCODE_BIN` (an `opencode` 2.x executable)."""

import asyncio
import itertools
import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import ValidationError

from verifiers.v1.harnesses.opencode.harness import (
    OpenCodeCompactionConfig,
    OpenCodeHarness,
    OpenCodeHarnessConfig,
)
from verifiers.v1.task import TaskData

SUMMARY = (
    "## Objective\n- count\n\n## Work State\n- counting\n\n## Next Steps\n- continue\n"
)


class Runtime:
    async def write(self, path: str, data: bytes) -> None:
        pass


async def opencode_env(endpoint: str, **config) -> dict[str, str]:
    harness = OpenCodeHarness(OpenCodeHarnessConfig(**config))
    acp = await harness.prepare_acp(
        SimpleNamespace(model="m"),
        SimpleNamespace(id="t"),
        Runtime(),
        endpoint,
        "k",
        {},
        TaskData(prompt="go"),
    )
    return acp.env


def settings(**config) -> dict:
    env = asyncio.run(opencode_env("http://e/v1", **config))
    return json.loads(env["OPENCODE_CONFIG_CONTENT"])


def test_compaction_sets_window_and_threshold():
    s = settings(compaction={"context_window": 262144, "summarize_at_tokens": 200000})
    assert s["providers"]["intercept"]["models"]["model"] == {
        "modelID": "m",
        # The window bounds output too, not OpenCode's default 32,000-token cap.
        "limit": {"context": 262144, "output": 262144},
    }
    assert s["compaction"] == {"buffer": 62144}


def test_window_alone_keeps_opencode_default_threshold():
    s = settings(compaction={"context_window": 262144})
    assert s["providers"]["intercept"]["models"]["model"]["limit"] == {
        "context": 262144,
        "output": 262144,
    }
    assert "compaction" not in s


def test_provider_output_cap_bounds_output():
    s = settings(
        compaction={
            "context_window": 1048576,
            "summarize_at_tokens": 917504,
            "max_output_tokens": 131072,
        }
    )
    assert s["providers"]["intercept"]["models"]["model"]["limit"] == {
        "context": 1048576,
        "output": 131072,
    }
    assert s["compaction"] == {"buffer": 131072}
    with pytest.raises(ValidationError):
        OpenCodeCompactionConfig(context_window=1000, max_output_tokens=2000)


def test_no_compaction_config_gives_no_window():
    s = settings()
    assert s["providers"]["intercept"]["models"]["model"] == {"modelID": "m"}
    assert "compaction" not in s


def test_threshold_must_be_below_window():
    with pytest.raises(ValidationError):
        OpenCodeCompactionConfig(context_window=1000, summarize_at_tokens=1000)


class Scripted(BaseHTTPRequestHandler):
    """Every turn runs one shell command with ~2k tokens of output; a summary request
    gets a summary. Prompt tokens are reported as len(json(messages)) / 4."""

    requests: ClassVar[list[dict]] = []
    turns = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(body)
        messages = body["messages"]
        prompt_tokens = len(json.dumps(messages)) // 4
        last = messages[-1]
        if last["role"] == "user" and "<template>" in str(last["content"]):
            deltas, finish = [{"role": "assistant", "content": SUMMARY}], "stop"
        else:
            type(self).turns += 1
            n = type(self).turns
            if n > 14:
                deltas, finish = [{"role": "assistant", "content": "Done."}], "stop"
            else:
                args = json.dumps(
                    {
                        "command": f"printf 'x%.0s' $(seq 8000); echo {n}",
                        "description": "x",
                    }
                )
                call = {"index": 0, "id": f"call_{n}", "type": "function"}
                deltas = [
                    {"role": "assistant", "reasoning_content": f"think {n}"},
                    {"content": f"Turn {n}."},
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
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 20,
                    "total_tokens": prompt_tokens + 20,
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
    not os.environ.get("VF_OPENCODE_BIN"), reason="needs VF_OPENCODE_BIN (opencode 2.x)"
)
def test_opencode_compacts_at_threshold_and_keeps_its_prefix():
    Scripted.requests, Scripted.turns = [], 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        with tempfile.TemporaryDirectory() as root:
            env = asyncio.run(
                opencode_env(
                    endpoint,
                    compaction={"context_window": 200000, "summarize_at_tokens": 12000},
                )
            )
            # The harness's state directories live under /tmp/vf-opencode-runs.
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
                timeout=300,
                check=True,
            )
    finally:
        server.shutdown()

    requests = Scripted.requests
    summaries = [
        i
        for i, r in enumerate(requests)
        if "<template>" in str(r["messages"][-1]["content"])
    ]
    assert summaries, "OpenCode never compacted"
    first = summaries[0]
    # Compaction fires at the threshold: within one ~2k-token turn of it, not before.
    tokens = [len(json.dumps(r["messages"])) // 4 for r in requests[:first]]
    assert all(t < 12000 for t in tokens)
    assert tokens[-1] >= 12000 - 2500
    # Before compaction each request extends the previous one exactly, so a rollout
    # is one branch; the turn after the summary starts from the checkpoint.
    for prev, cur in itertools.pairwise(requests[:first]):
        assert cur["messages"][: len(prev["messages"])] == prev["messages"]
        assert cur["tools"] == prev["tools"]
    resumed = requests[first + 1]["messages"]
    assert "<conversation-checkpoint>" in resumed[1]["content"]
    assert SUMMARY.strip() in resumed[1]["content"]
    assert Scripted.turns > 14  # the rollout ran to its end after compacting


def test_location_idle_timeout_reaches_opencode_settings():
    """OpenCode 2.0 unloads a workspace (interrupting its session) after an hour without
    durable activity, which a single long model turn is; 2.1 takes the limit, or False
    to never unload, from experimental.location_idle_timeout."""
    assert "experimental" not in settings()
    assert settings(location_idle_timeout=False)["experimental"] == {
        "location_idle_timeout": False
    }
    assert settings(location_idle_timeout="24 hours")["experimental"] == {
        "location_idle_timeout": "24 hours"
    }


def test_input_modalities_declare_a_text_only_model():
    """A text-only model must say so: OpenCode otherwise attaches an image the agent reads,
    and GLM 5.3 rejects the request ("Unsupported input modality: image"), ending the
    rollout. OpenCode 2 reads `capabilities.input`; it drops a `modalities` key."""
    assert "capabilities" not in settings()["providers"]["intercept"]["models"]["model"]
    model = settings(input_modalities=["text"])["providers"]["intercept"]["models"][
        "model"
    ]
    assert model["capabilities"] == {"input": ["text"], "output": ["text"]}
    assert "modalities" not in model


# A 1x1 PNG.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAAB"
    "JRU5ErkJggg=="
)


class ReadsAnImage(BaseHTTPRequestHandler):
    """The agent writes a PNG with the shell, reads it, then stops."""

    requests: ClassVar[list[dict]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(body)
        script = [
            (
                "shell",
                {
                    "command": f"echo {PNG} | base64 -d > logo.png",
                    "description": "x",
                },
            ),
            ("read", {"path": "logo.png"}),
        ]
        n = len(type(self).requests)
        if n <= len(script):
            name, args = script[n - 1]
            call = {"index": 0, "id": f"call_{n}", "type": "function"}
            deltas = [
                {"role": "assistant", "content": f"Turn {n}."},
                {
                    "tool_calls": [
                        {
                            **call,
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ]
                },
            ]
            finish = "tool_calls"
        else:
            deltas, finish = [{"role": "assistant", "content": "Done."}], "stop"
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
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def run_reads_an_image(**config) -> list[dict]:
    ReadsAnImage.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReadsAnImage)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        with tempfile.TemporaryDirectory() as root:
            env = asyncio.run(opencode_env(endpoint, **config))
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
                timeout=300,
                check=True,
            )
    finally:
        server.shutdown()
    return ReadsAnImage.requests


@pytest.mark.skipif(
    not os.environ.get("VF_OPENCODE_BIN"), reason="needs VF_OPENCODE_BIN (opencode 2.x)"
)
def test_text_only_model_never_receives_the_image_it_reads():
    """GLM nl2repo draws wrote a PNG and read it back to look at it; with images on,
    the next request carried an image_url part and the provider rejected it."""
    sent = json.dumps(run_reads_an_image()[-1]["messages"])
    assert "data:image/png;base64" in sent
    requests = run_reads_an_image(input_modalities=["text"])
    assert len(requests) == 3
    sent = json.dumps(requests[-1]["messages"])
    assert "image_url" not in sent and "data:image" not in sent
    assert "this model does not support image input" in sent
    # Tools stay on.
    assert {t["function"]["name"] for t in requests[-1]["tools"]} >= {"read", "shell"}
