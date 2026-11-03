"""Rollout checkpoints on Prime: a model call fails past its turn retries, the rollout
checkpoints its box, and the rollout retry restores it. A memory checkpoint resumes the
box with its processes: a background server, a REPL variable and a persistent
sub-agent keep their state. A filesystem checkpoint restores the files into a new box,
whose processes are gone.

The model is scripted: it fails one turn with 503s until the turn retries run out,
then answers the replayed turn."""

import json
import os

import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.configs.agent import (
    MemoryCheckpointConfig,
    RolloutCheckpointConfig,
    TurnRetryConfig,
)
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.harnesses.bash import BashHarnessConfig
from verifiers.v1.harnesses.rlm.harness import RLMHarnessConfig
from verifiers.v1.utils.prime import load_prime_config

pytestmark = pytest.mark.skipif(
    not (os.environ.get("PRIME_API_KEY") or load_prime_config().get("api_key")),
    reason="needs Prime credentials",
)

COUNTER = (
    'nohup python3 -c "import http.server as h\nn=[0]\n'
    "class H(h.BaseHTTPRequestHandler):\n"
    " def do_GET(s):\n  n[0]+=1; s.send_response(200); s.end_headers(); "
    "s.wfile.write(str(n[0]).encode())\n"
    " def log_message(s,*a): pass\n"
    "h.HTTPServer(('127.0.0.1',9999),H).serve_forever()\" >/dev/null 2>&1 & sleep 1; "
)
READ_COUNTER = (
    'python3 -c "import urllib.request;'
    "print('counter', urllib.request.urlopen('http://127.0.0.1:9999').read().decode())\""
)
SPAWN = """import os
child = await rlm.agent.spawn(task="SUBAGENT: remember the number 7", name="keeper", persistent=True)
first = await child.result(yield_after=120)
kept = 41
print('pid', os.getpid(), 'kept', kept, 'child', first.answer)
"""
ASK = """import asyncio, os
await child.send("SUBAGENT: what number did you remember?")
answer = None
for _ in range(120):
    if (await child.info()).status == "idle":
        result = await child.result(yield_after=1)
        if result.answer != "noted":
            answer = result.answer
            break
    await asyncio.sleep(1)
print('pid', os.getpid(), 'kept + 1 =', kept + 1, 'child', answer)
"""


def _text(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def _tool_call(name: str, arguments: dict, n: int) -> dict:
    call = {"name": name, "arguments": json.dumps(arguments)}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": f"call_{n}", "type": "function", "function": call}],
    }


@pytest.fixture
async def outage_model():
    """A chat-completions model that plays `script` (one reply per root turn) and fails
    the second root turn with 503s `failures` times. A request whose prompt holds
    "SUBAGENT: remember" is a sub-agent's: it answers "noted", then "7" once asked."""
    plan: dict = {}

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        messages = body["messages"]
        prompts = [_text(m) for m in messages if m["role"] in ("user", "system")]
        if any("SUBAGENT: remember" in p for p in prompts):
            asked = any("SUBAGENT: what number" in p for p in prompts)
            message = {"role": "assistant", "content": "7" if asked else "noted"}
        else:
            turn = sum(1 for m in messages if m["role"] == "assistant")
            if turn == 1 and plan["failures"]:
                plan["failures"] -= 1
                return web.json_response({"error": {"message": "outage"}}, status=503)
            script = plan["script"]
            message = script[turn] if turn < len(script) else script[-1]
        raw = {
            "id": "chatcmpl-scripted",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "scripted"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                    if "tool_calls" in message
                    else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        dialect = ChatDialect()
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

    def play(script: list[dict], failures: int = 2) -> vf.EvalClientConfig:
        plan.update(script=script, failures=failures)
        return vf.EvalClientConfig(base_url=f"http://127.0.0.1:{port}/v1")

    try:
        yield play
    finally:
        await runner.cleanup()


async def _run(harness, client, *, memory: bool) -> vf.Trace:
    agent = vf.make_agent(
        vf.AgentConfig(
            harness=harness,
            runtime=vf.PrimeConfig(image="python:3.11-slim", labels=["vf-ci"]),
            model="scripted",
            client=client,
            turn_retries=TurnRetryConfig(max_retries=1, initial_delay=0.1),
            checkpoint=RolloutCheckpointConfig(),
            memory_checkpoint=MemoryCheckpointConfig() if memory else None,
        )
    )
    return await agent.run(vf.Task(vf.TaskData(idx=0, prompt="Go.")))


def _restored(trace: vf.Trace, *, memory: bool) -> None:
    assert trace.ok, [error.message for error in trace.errors]
    assert [error.type for error in trace.errors] == ["ProviderError"]
    (checkpoint,) = trace.checkpoints
    assert checkpoint.reason == "ProviderError"
    assert checkpoint.restored and checkpoint.memory is memory


def _tool_results(trace: vf.Trace) -> list[str]:
    return [str(m.content) for m in trace.messages if m.role == "tool"]


@pytest.mark.parametrize("memory", [True, False], ids=["memory", "filesystem"])
async def test_bash_background_process(outage_model, memory):
    """The background server answers again after a memory restore; a filesystem
    restore boots a new box without it."""
    script = [
        _tool_call("bash", {"command": COUNTER + READ_COUNTER}, 0),
        _tool_call("bash", {"command": READ_COUNTER}, 1),
        {"role": "assistant", "content": "done"},
    ]
    trace = await _run(
        BashHarnessConfig(id="bash"), outage_model(script), memory=memory
    )
    _restored(trace, memory=memory)
    first, second = _tool_results(trace)
    assert first.strip() == "counter 1"
    if memory:
        assert second.strip() == "counter 2"
    else:
        assert "Connection refused" in second


@pytest.mark.skipif(
    not hasattr(__import__("prime_sandboxes").AsyncSandboxClient, "attach_process"),
    reason="needs a prime-sandboxes with attach_process",
)
async def test_rlm_kernel_and_sub_agent(outage_model):
    """rlm over ACP: the REPL kernel and a persistent sub-agent keep their state
    through a memory restore; the new host attaches to the running ACP runner."""
    script = [
        _tool_call("ipython", {"code": SPAWN}, 0),
        _tool_call("ipython", {"code": ASK}, 1),
        {"role": "assistant", "content": "done"},
    ]
    trace = await _run(RLMHarnessConfig(id="rlm"), outage_model(script), memory=True)
    _restored(trace, memory=True)
    first, second = _tool_results(trace)
    pid = first.split()[1]
    assert first.split()[2:] == ["kept", "41", "child", "noted"]
    assert second.split() == ["pid", pid, "kept", "+", "1", "=", "42", "child", "7"]
