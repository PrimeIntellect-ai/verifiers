"""The in-box ACP runner ends a turn whose agent stalled without ending it.

OpenCode never ends its turn once a write to its database fails: on ProgramBench boxes
whose 5 GB disk filled (2026-10-28) it took a tool call's reply and then neither sent
the tool's result to the model nor answered the prompt, while the runner kept beating
its heartbeat, so six rollouts held their slots until cancelled. The runner sees the
agent's model requests (its relay) and tool calls (ACP updates): a turn idle on a full
disk, or idle with nothing open, is cancelled and raises `AgentStalled`.

RLM shows neither: its provider comes from session metadata the relay doesn't rewrite,
and it sends no tool call updates. On 2026-10-28/29 every CyberStrike RLM turn still
working 600 s in (one made 229 model calls, the last 1 s before) was failed as stalled.
Idleness counts only for what the runner has seen the agent do."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import verifiers.v1.acp as acp_harness


@pytest.fixture
def runner(monkeypatch):
    """runner.py is a uv script run inside the box; its `acp` dependency is not installed
    here, and the watchdog doesn't use it, so a stub stands in."""
    stub = types.ModuleType("acp")
    for name in (
        "PROTOCOL_VERSION",
        "Client",
        "RequestError",
        "image_block",
        "spawn_agent_process",
        "text_block",
    ):
        setattr(
            stub, name, type(name, (Exception,) if name == "RequestError" else (), {})
        )
    stub.text_block = lambda text: {"type": "text", "text": text}
    schema = types.ModuleType("acp.schema")
    schema.__getattr__ = lambda name: type(name, (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acp", stub)
    monkeypatch.setitem(sys.modules, "acp.schema", schema)
    path = Path(acp_harness.__file__).parent / "runner.py"
    spec = importlib.util.spec_from_file_location("vf_acp_runner_stall", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WATCH_SECONDS", 0.01)
    monkeypatch.setattr(module, "DISK_FULL_IDLE_SECONDS", 0.1)
    monkeypatch.setattr(module, "AGENT_IDLE_SECONDS", 0.3)
    return module


def tool(kind: str, status: str | None, tool_call_id: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(
        session_update=kind, tool_call_id=tool_call_id, status=status
    )


def seen(runner, *, tools: bool = True):
    """An agent whose model requests come through the relay, and that reports its tool
    calls (OpenCode)."""
    activity = runner.Activity()
    activity.sees_requests = True
    activity.sees_tools = tools
    return activity


def test_activity_tracks_open_tool_calls(runner):
    activity = runner.Activity()
    activity.update(tool("tool_call", "pending"))
    activity.update(tool("tool_call_update", "in_progress"))
    assert activity.tools == {"t1"}
    activity.update(tool("tool_call_update", None))  # a content-only update
    assert activity.tools == {"t1"}
    activity.update(tool("tool_call_update", "completed"))
    assert activity.tools == set()
    activity.update(SimpleNamespace(session_update="agent_message_chunk"))
    assert activity.tools == set()


async def test_a_full_disk_stalls_an_idle_agent_even_mid_tool(runner, monkeypatch):
    monkeypatch.setattr(runner, "free_bytes", lambda path: 4096)
    activity = seen(runner)
    activity.update(tool("tool_call", "in_progress"))
    stalled = await asyncio.wait_for(runner.watch(activity, "/data"), 5)
    assert stalled.kind == "disk_full" and "4096 bytes free" in str(stalled)


async def test_an_idle_agent_with_nothing_open_is_stalled(runner, monkeypatch):
    monkeypatch.setattr(runner, "free_bytes", lambda path: 10**9)
    stalled = await asyncio.wait_for(runner.watch(seen(runner), "/data"), 5)
    assert stalled.kind == "agent_stalled"


async def test_an_agent_the_runner_cannot_see_is_never_judged_idle(runner, monkeypatch):
    """RLM: no request through the relay, no tool call update. Its model request or
    tool run in flight looks like nothing, so silence says nothing about it."""
    monkeypatch.setattr(runner, "free_bytes", lambda path: 10**9)
    rlm = runner.Activity()
    rlm.update(SimpleNamespace(session_update="agent_message_chunk"))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.watch(rlm, "/data"), 0.6)
    monkeypatch.setattr(runner, "free_bytes", lambda path: 0)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.watch(runner.Activity(), "/data"), 0.6)


async def test_an_agent_that_reports_no_tool_calls_is_stalled_only_on_a_full_disk(
    runner, monkeypatch
):
    """Its requests are seen, but a tool run is not: idle with nothing seen open may
    be a long command, though a full disk still ends it."""
    monkeypatch.setattr(runner, "free_bytes", lambda path: 10**9)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.watch(seen(runner, tools=False), "/data"), 0.6)
    monkeypatch.setattr(runner, "free_bytes", lambda path: 0)
    stalled = await asyncio.wait_for(
        runner.watch(seen(runner, tools=False), "/data"), 5
    )
    assert stalled.kind == "disk_full"


def test_activity_learns_what_the_agent_shows(runner):
    activity = runner.Activity()
    assert not activity.sees_requests and not activity.sees_tools
    activity.update(SimpleNamespace(session_update="agent_message_chunk"))
    assert not activity.sees_tools
    activity.update(tool("tool_call", "pending"))
    assert activity.sees_tools
    activity.request()
    assert activity.sees_requests and activity.requests == 1


async def test_a_model_request_or_an_open_tool_is_never_a_stall(runner, monkeypatch):
    """A generation can run for hours, and a command as long as the model allows."""
    monkeypatch.setattr(runner, "free_bytes", lambda path: 10**9)
    generating = seen(runner)
    generating.requests = 1
    running = seen(runner)
    running.update(tool("tool_call", "in_progress"))
    for activity in (generating, running):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(runner.watch(activity, "/data"), 0.6)


class WedgedConnection:
    """An agent that took the prompt and never answers it."""

    def __init__(self) -> None:
        self.cancelled = False

    async def prompt(self, **kwargs):
        await asyncio.Event().wait()

    async def cancel(self, **kwargs):
        self.cancelled = True


async def test_a_watched_turn_whose_agent_wedges_raises_agent_stalled(
    runner, monkeypatch
):
    monkeypatch.setattr(runner, "free_bytes", lambda path: 0)
    client = runner.VerifiersACPClient()
    client.activity.sees_requests = client.activity.sees_tools = True
    connection = WedgedConnection()
    config = {"system_prompt": "", "user_contents": ["go"]}
    with pytest.raises(runner.AgentStalled) as raised:
        await asyncio.wait_for(
            runner.prompt(
                client, connection, None, "s", config, is_new=True, watched="/data"
            ),
            5,
        )
    assert raised.value.kind == "disk_full"
    assert connection.cancelled and client.prompt_task is None


async def test_the_relay_counts_model_requests_in_flight(runner):
    """A request the relay is forwarding is the agent at work, however long it takes."""
    import httpx
    from aiohttp import web

    release = asyncio.Event()

    async def slow(request: web.Request) -> web.Response:
        await release.wait()
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", slow)
    upstream = web.AppRunner(app)
    await upstream.setup()
    site = web.TCPSite(upstream, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    activity = runner.Activity()
    relay = runner.ProviderRelay(f"http://127.0.0.1:{port}/v1", activity)
    await relay.start()
    try:
        async with httpx.AsyncClient() as client:
            call = asyncio.create_task(
                client.post(f"{relay.local_origin}/v1/chat/completions", json={})
            )
            for _ in range(200):
                if activity.requests:
                    break
                await asyncio.sleep(0.01)
            assert activity.requests == 1 and activity.sees_requests
            release.set()
            assert (await call).json() == {"ok": True}
            for _ in range(200):
                if not activity.requests:
                    break
                await asyncio.sleep(0.01)
            assert activity.requests == 0
    finally:
        await relay.close()
        await upstream.cleanup()
