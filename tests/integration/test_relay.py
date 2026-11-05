"""The interception relay on Prime: the server's tunnel is lost while a rollout's model
call is in flight: its registration is deleted and its frpc connection drops, so it
cannot log in again. With `agent.relay`, the server opens a new tunnel, reroutes the box's
relay to it, and the rollout finishes as if nothing happened: the waiting model call
goes through once, on one branch. Without the relay, the rollout fails, as before.

The model is scripted and holds its second reply until the test has dropped the tunnel
and, with the relay, until the server has replaced it."""

import asyncio
import json
import os
import signal
from typing import Self

import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.harnesses.bash import BashHarnessConfig
from verifiers.v1.interception import InterceptionServer
from verifiers.v1.utils.prime import load_prime_config

pytestmark = pytest.mark.skipif(
    not (os.environ.get("PRIME_API_KEY") or load_prime_config().get("api_key")),
    reason="needs Prime credentials",
)

DETECTION_SECONDS = 180
"""How long the tunnel's health probe may take to declare the lost tunnel gone, plus the
time to open a new one."""


def _tool_call(command: str, n: int) -> dict:
    call = {"name": "bash", "arguments": json.dumps({"command": command})}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": f"call_{n}", "type": "function", "function": call}],
    }


SCRIPT = [
    _tool_call("echo before-the-tunnel-loss > /tmp/note", 0),
    _tool_call("cat /tmp/note", 1),
    {"role": "assistant", "content": "done"},
]


class HeldModel:
    """A chat-completions model that plays `SCRIPT`, one reply per turn, and holds the
    reply to turn 1 until `release` is set. It counts the requests per turn."""

    def __init__(self) -> None:
        self.requests: dict[int, int] = {}
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self.url = ""
        self.runner: web.AppRunner | None = None

    async def answer(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        turn = sum(1 for m in body["messages"] if m["role"] == "assistant")
        self.requests[turn] = self.requests.get(turn, 0) + 1
        if turn == 1:
            self.held.set()
            await self.release.wait()
        message = SCRIPT[min(turn, len(SCRIPT) - 1)]
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

    async def __aenter__(self) -> Self:
        app = web.Application()
        app.router.add_post("/{tail:.*}", self.answer)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
        return self

    async def __aexit__(self, *exc) -> None:
        self.release.set()
        assert self.runner is not None
        await self.runner.cleanup()


async def drop_tunnel(url: str) -> None:
    """Lose the tunnel that serves `url` for good: delete its registration, then end its
    frpc connection, which then cannot log in again. A deleted registration alone keeps
    serving while frpc stays connected."""
    from prime_tunnel import TunnelClient

    async with TunnelClient() as api:
        tunnels = await api.list_tunnels(per_page=100)
        (tunnel,) = [t for t in tunnels if t.url.rstrip("/") == url.rstrip("/")]
        assert await api.delete_tunnel(tunnel.tunnel_id)
    frpc = [pid for pid in _children() if f"{tunnel.tunnel_id}.toml" in _cmdline(pid)]
    assert frpc, "no frpc process for the tunnel"
    for pid in frpc:
        os.kill(pid, signal.SIGKILL)


def _children() -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stat:
                ppid = int(stat.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == os.getpid():
            pids.append(int(entry))
    return pids


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline") as cmdline:
            return cmdline.read()
    except OSError:
        return ""


async def _run(relay: bool) -> tuple[vf.Trace, HeldModel, str, str]:
    async with HeldModel() as model, InterceptionServer(requires_tunnel=True) as server:
        agent = vf.make_agent(
            vf.AgentConfig(
                harness=BashHarnessConfig(id="bash"),
                runtime=vf.PrimeConfig(image="python:3.11-slim", labels=["vf-ci"]),
                model="scripted",
                client=vf.EvalClientConfig(base_url=model.url),
                relay=relay,
            ),
            interception=server,
        )
        run = asyncio.create_task(agent.run(vf.Task(vf.TaskData(idx=0, prompt="Go."))))
        await asyncio.wait_for(model.held.wait(), 600)
        first_url = server.base_url
        await drop_tunnel(first_url)
        if relay:
            # Hold the reply until the server serves the rollout through a new tunnel.
            async with asyncio.timeout(DETECTION_SECONDS):
                while server.base_url == first_url:
                    await asyncio.sleep(1)
        else:
            async with asyncio.timeout(DETECTION_SECONDS):
                while server.healthy:
                    await asyncio.sleep(1)
        model.release.set()
        trace = await asyncio.wait_for(run, 600)
        return trace, model, first_url, server.base_url


async def test_rollout_survives_a_replaced_tunnel():
    trace, model, first_url, last_url = await _run(relay=True)

    assert last_url != first_url
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.errors == []
    tool_results = [str(m.content) for m in trace.messages if m.role == "tool"]
    assert tool_results[-1].strip() == "before-the-tunnel-loss"
    # The held call reached the model once: the relay's retry joined the request the
    # server was already serving.
    assert model.requests[1] == 1
    assert len(trace.branches) == 1


async def test_rollout_without_relay_fails_on_a_lost_tunnel():
    trace, _, _, _ = await _run(relay=False)

    assert not trace.ok
    # The harness gets the tunnel edge's 404 and exits, usually before the server
    # declares the tunnel gone and fails the rollout itself.
    assert {error.type for error in trace.errors} & {"HarnessError", "TunnelError"}
