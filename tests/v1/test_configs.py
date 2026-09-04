"""Every checked-in v1 eval config parses, and the eval CLI says what a config resolved
to before it runs anything.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig).
The CLI cases take the model-free path — the `echo-v1` fixture on the null harness in a
subprocess, against a local stub endpoint — so they need no key and no sandbox.
"""

import json
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from verifiers.v1.configs.cli.eval import EvalConfig

CONFIGS = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


class EchoModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint whose every chat completion is the last user
    message — echo-v1 scores that 1.0."""

    def log_message(self, *args) -> None:  # keep the access log off the test output
        pass

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reply = next(
            m["content"] for m in reversed(request["messages"]) if m["role"] == "user"
        )
        body = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_endpoint():
    """A local `EchoModel` for the run's client; its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), EchoModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def eval_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """`uv run eval <args>` under the test interpreter (the fixture tasksets resolve
    through the `PYTHONPATH` conftest sets)."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from verifiers.v1.cli.eval.main import main; main()",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_eval_names_where_it_runs(stub_endpoint: str, tmp_path: Path) -> None:
    """Before the first rollout starts, the log names each seat's harness and runtime,
    each client's endpoint and credential variable (the name, never the value), and
    whether the finished run is pushed — so a seat the config left on the defaults is
    visible up front."""
    proc = eval_cli(
        "echo-v1",
        "--env.agent.harness.id", "null",
        "--env.agent.runtime.type", "subprocess",
        "-m", "stub-model",
        "--client.base-url", stub_endpoint,
        "--client.api-key-var", "VF_TEST_KEY",
        "--no-push", "--no-rich", "-n", "1", "-r", "1",
        "-o", str(tmp_path), "--run.name", "run",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    log = proc.stderr
    assert "rollout start" in log, log
    started = log.index("rollout start")
    for line in (
        "env.agent: null harness on the subprocess runtime",
        f"client: {stub_endpoint} ($VF_TEST_KEY)",
        "push: off",
    ):
        assert line in log, log
        assert log.index(line) < started, log


def test_dry_run_prints_resolved_config(tmp_path: Path) -> None:
    """`--dry-run` prints the resolved config it writes: the same JSON document, naming
    the role's runtime and the client's endpoint and credential variable."""
    proc = eval_cli(
        "echo-v1",
        "--client.base-url", "http://127.0.0.1:9/v1",
        "--client.api-key-var", "VF_TEST_KEY",
        "--dry-run", "-o", str(tmp_path), "--run.name", "run",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    printed = json.loads(proc.stdout)
    written = json.loads((tmp_path / "run/configs/resolved/eval.json").read_text())
    assert printed == written
    assert "type" in printed["env"]["agent"]["runtime"]
    assert printed["client"]["base_url"] == "http://127.0.0.1:9/v1"
    assert printed["client"]["api_key_var"] == "VF_TEST_KEY"
