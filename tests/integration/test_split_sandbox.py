"""Split harness execution and the proc, file, and control-plane boundary."""

import asyncio
import json
import os
import secrets
import shutil
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from conftest import _eval_config
from grid import HARNESSES, grid

from verifiers.v1.cli.run import run_episodes
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.harnesses.bash.harness import BashHarness, BashHarnessConfig
from verifiers.v1.runtimes.deployment import Deployment
from verifiers.v1.types import ToolMessage
from verifiers.v1.utils.prime import load_prime_config


@pytest.mark.parametrize(
    "harness,harness_runtime",
    grid([*HARNESSES, {"id": "kimi-code", "dialect": "responses"}, "terminus-2"]),
    indirect=True,
)
async def test_split_harness_workspace(run, harness, harness_runtime, monkeypatch):
    deployments = []
    local_files = []
    original_start = Deployment.start
    original_close = Deployment.close

    async def capture_start(self):
        await original_start(self)
        deployments.append(self)

    async def inspect_then_close(self):
        try:
            if self in deployments:
                result = await self.harness_backend.bind().run(
                    ["sh", "-c", "test ! -e answer.txt"], {}
                )
                local_files.append(result.exit_code == 0)
        finally:
            await original_close(self)

    monkeypatch.setattr(Deployment, "start", capture_start)
    monkeypatch.setattr(Deployment, "close", inspect_then_close)
    (episode,) = await run(
        "echo" if harness.id == "null" else "echo-agentic",
        harness=harness,
        runtime=harness_runtime,
        env={"agent": {"deployment": {"mode": "split"}}},
        max_turns=10,
        max_tokens=8192,
        rollout_timeout=300,
    )
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert local_files and all(local_files), "Tool also wrote into the harness sandbox"
    for deployment in deployments:
        assert deployment.harness_backend is not deployment.execution_backend
        assert (
            deployment.harness_backend.info.id != deployment.execution_backend.info.id
        )
        await deployment.harness_backend.verify_stopped()
        await deployment.execution_backend.verify_stopped()


def _probe_command(control_url: str) -> str:
    return r"""python3 - <<'PY'
import glob, json, os, re, urllib.error, urllib.request

found = set()
pattern = re.compile(rb'VF_SPLIT_(?:ENV|MEMORY)_SENTINEL_[0-9a-f]{32}')
def scan(data):
    found.update(match.decode() for match in pattern.findall(data))

def read(path, offset=0):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            os.lseek(fd, offset, os.SEEK_SET)
            scan(os.read(fd, 32768))
        finally:
            os.close(fd)
    except (OSError, ValueError):
        pass

for pid in glob.glob('/proc/[0-9]*'):
    for leaf in ('environ', 'cmdline', 'mem'):
        read(f'{pid}/{leaf}')
    for fd in glob.glob(f'{pid}/fd/*')[:64]:
        try:
            scan(os.readlink(fd).encode())
        except OSError:
            pass
        read(fd)
    try:
        with open(f'{pid}/maps') as maps:
            for line in list(maps)[:64]:
                if 'r' in line.split()[1]:
                    read(f'{pid}/mem', int(line.split('-')[0], 16))
    except (OSError, IndexError, ValueError):
        pass

count = 0
for root in ('/app', '/tmp', '/run', '/root'):
    for directory, _, files in os.walk(root):
        for name in files:
            read(os.path.join(directory, name))
            count += 1
            if count >= 200:
                break
        if count >= 200:
            break
    if count >= 200:
        break

try:
    urllib.request.urlopen(CONTROL_URL, timeout=3).read(32)
    control = 'reachable'
except urllib.error.HTTPError:
    control = 'reachable'
except (OSError, urllib.error.URLError):
    control = 'blocked'
print('PROBE=' + json.dumps({'found': sorted(found), 'control': control}))
PY""".replace("CONTROL_URL", repr(f"{control_url.rstrip('/')}/v1/models"))


@pytest.mark.parametrize("provider", ["docker", "prime"])
async def test_split_bash_boundary(tmp_path, monkeypatch, provider):
    if provider == "docker":
        if shutil.which("docker") is None:
            pytest.skip("Docker CLI or daemon unavailable")
        probe = await asyncio.create_subprocess_exec(
            "docker",
            "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await probe.wait():
            pytest.skip("Docker CLI or daemon unavailable")
    elif not (os.environ.get("PRIME_API_KEY") or load_prime_config().get("api_key")):
        pytest.skip("Prime sandbox credentials unavailable")

    env_sentinel = f"VF_SPLIT_ENV_SENTINEL_{secrets.token_hex(16)}"
    memory_sentinel = f"VF_SPLIT_MEMORY_SENTINEL_{secrets.token_hex(16)}"
    control_urls: list[str] = []
    backends = []
    workers = []
    dialect = ChatDialect()

    original_start = Deployment.start

    async def capture_start(self):
        await original_start(self)
        backends.extend([self.harness_backend, self.execution_backend])

    monkeypatch.setattr(Deployment, "start", capture_start)

    original_setup = BashHarness.setup

    async def setup_with_memory(self, runtime):
        await original_setup(self, runtime)
        worker = await runtime.exec(
            [
                "python3",
                "-u",
                "-c",
                (
                    "import sys,time; secret=bytearray(sys.stdin.buffer.readline()); "
                    "print('READY',flush=True); time.sleep(600)"
                ),
            ],
            {},
        )
        await worker.write((memory_sentinel + "\n").encode())
        async with asyncio.timeout(10):
            assert b"READY" in await anext(worker.stdout)
        workers.append(worker)

    monkeypatch.setattr(BashHarness, "setup", setup_with_memory)

    import verifiers.v1.rollout as rollout_module

    original_serve = rollout_module.serve_interception

    @asynccontextmanager
    async def capture_interception(*args, **kwargs):
        async with original_serve(*args, **kwargs) as slot:
            control_urls.append(slot[0])
            yield slot

    monkeypatch.setattr(rollout_module, "serve_interception", capture_interception)

    async def answer(request):
        body = await request.json()
        turn = sum(message.get("role") == "tool" for message in body["messages"])
        if turn == 0:
            name, arguments = "bash", {"command": "touch blocked.txt"}
        elif turn == 1:
            name, arguments = "bash", {"command": "printf 'before\\n' > work.txt"}
        elif turn == 2:
            name, arguments = (
                "edit",
                {"path": "work.txt", "old_str": "before", "new_str": "after"},
            )
        elif turn == 3:
            name, arguments = "bash", {"command": _probe_command(control_urls[-1])}
        else:
            name = ""
            arguments = {}
        message = {"role": "assistant", "content": "done" if not name else None}
        if name:
            message["tool_calls"] = [
                {
                    "id": f"split-{turn}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        raw = {
            "id": f"chatcmpl-split-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if name else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        config = _eval_config(
            "split-sandbox",
            harness=BashHarnessConfig(
                id="bash", env={"VF_SPLIT_ENV_SENTINEL": env_sentinel}
            ),
            runtime={"type": provider},
            env={"agent": {"deployment": {"mode": "split"}}},
            client={"base_url": f"http://127.0.0.1:{port}/v1"},
            max_turns=6,
        )
        config.output_dir = tmp_path
        (episode,) = await run_episodes(config)
    finally:
        await runner.cleanup()

    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    results = [
        message for message in trace.messages if isinstance(message, ToolMessage)
    ]
    assert [message.tool_call_id for message in results] == [
        f"split-{i}" for i in range(4)
    ]
    assert results[0].content == "POLICY_BLOCKED"
    assert "PROBE=" in results[-1].content
    report = json.loads(results[-1].content.split("PROBE=", 1)[1].strip())
    assert report == {"found": [], "control": "blocked"}
    assert env_sentinel not in str(results) and memory_sentinel not in str(results)
    assert len(backends) == 2 and backends[0] is not backends[1]
    assert workers
    for backend in backends:
        await backend.verify_stopped()
