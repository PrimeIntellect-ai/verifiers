import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from environments.mcp_fetch.mcp_env import DEFAULT_FIXTURE_PORT, load_environment
from environments.mcp_fetch.tools.fetch_mcp_server import fetch_url_async
from environments.mcp_fetch.verifiers import run_verifier

TASKS_PATH = Path("environments/mcp_fetch/tasks/qa.jsonl").resolve()


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_port_state(port: int, expected_open: bool, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_in_use(port) == expected_open:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Port {port} did not reach state={expected_open}")


def _load_tasks() -> dict[str, dict]:
    data: dict[str, dict] = {}
    with TASKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data[str(row["id"])] = row
    return data


@pytest.fixture
def fixture_server():
    """Ensure the deterministic fixture server is running for a test."""

    port = DEFAULT_FIXTURE_PORT
    if _port_in_use(port):
        yield None
        return

    cmd = [
        sys.executable,
        "-m",
        "environments.mcp_fetch.utils.mini_httpd",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _wait_port_state(port, expected_open=True)
    yield proc
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    _wait_port_state(port, expected_open=False)


def test_fetch_env_auto_starts_fixture():
    if _port_in_use(DEFAULT_FIXTURE_PORT):
        pytest.skip("Fixture port already in use; skipping auto-start test")
    env = load_environment()
    assert env.dataset is not None
    _wait_port_state(DEFAULT_FIXTURE_PORT, expected_open=True)
    env.close()
    _wait_port_state(DEFAULT_FIXTURE_PORT, expected_open=False)


@pytest.mark.asyncio
async def test_fetch_tool_verifiers_cover_core_cases(fixture_server):
    tasks = _load_tasks()
    base_url = f"http://127.0.0.1:{DEFAULT_FIXTURE_PORT}"
    shard_cases = [
        ("fetch_014", {"path": "/text/poem.txt", "method": "GET"}),
        ("fetch_006", {"path": "/html/about.html", "method": "HEAD"}),
        ("fetch_009", {"path": "/query?category=fruits&limit=2"}),
        ("fetch_011", {"path": "/auth", "headers": {"X-Token": "opensesame"}}),
        ("fetch_018", {"path": "/json/data_large.jsonl"}),
        ("fetch_027", {"path": "/json/data_large.jsonl"}),
        ("fetch_030", {"path": "/json/ledger.json"}),
    ]
    for task_id, request in shard_cases:
        req = dict(request)
        rel_path = req.pop("path")
        req["url"] = f"{base_url}{rel_path}"
        payload = await fetch_url_async(**req)
        verifier = tasks[task_id]["verifier"]
        result = run_verifier(verifier, payload)
        assert result is True


def test_fetch_env_dataset_and_parser(fixture_server):
    env = load_environment(auto_start_fixtures=False)
    assert env.dataset is not None
    assert len(env.dataset) >= 35
    assert env.host_allowlist[0].startswith("127.0.0.1")

    prompt = env.dataset[0]["prompt"]
    completion = [{"role": "assistant", "content": "ANSWER: test"}]
    state = {
        "prompt": prompt,
        "completion": completion,
        "responses": [],
        "turn": 0,
        "timing": {"generation_ms": 0.0, "total_ms": 0.0, "scoring_ms": 0.0},
        "task": "default",
        "info": {},
    }
    score = asyncio.run(
        env.rubric.score_rollout(prompt, completion, env.dataset[0]["answer"], state)
    )
    assert "accuracy" in score.metrics
    env.close()
