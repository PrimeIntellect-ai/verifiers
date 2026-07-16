"""Agent guards that fire before any model or runtime I/O (no live calls)."""

from unittest.mock import patch

import pytest

import verifiers.v1 as vf
import verifiers.v1.agent
from verifiers.v1.agent import _check_borrowed_placement
from verifiers.v1.harnesses.default import DefaultHarness, DefaultHarnessConfig
from verifiers.v1.runtimes import DockerConfig, SubprocessConfig, make_runtime


def _agent() -> vf.Agent:
    # The guard under test raises before the model context is touched, so the
    # client can be a stub.
    ctx = vf.ModelContext(model="test-model", client=None)  # type: ignore[arg-type]
    return vf.Agent(DefaultHarness(DefaultHarnessConfig()), ctx)


async def test_borrowed_subprocess_box_refuses_task_image():
    """Parity with the provisioning path: `resolve_runtime_config` refuses a task
    `image` on the subprocess runtime, and the borrowed branch never resolves — so
    the borrowed guard must refuse the same pairing."""
    box = make_runtime(SubprocessConfig())
    task = vf.Task(vf.TaskData(idx=0, prompt="hi", image="python:3.12-slim"))
    with pytest.raises(ValueError, match="requires image"):
        await _agent().run(task, runtime=box)


def test_borrowed_container_box_with_other_image_warns():
    """A container box whose image differs from the task's can still be the point of
    borrowing (a judge in a solver's world) — signal, don't refuse."""
    box = make_runtime(DockerConfig(image="ubuntu:24.04"))
    task = vf.Task(vf.TaskData(idx=0, prompt="hi", image="python:3.12-slim"))
    # The `verifiers` package logger doesn't propagate to root, so caplog never sees
    # these records; assert on the module logger itself.
    with patch.object(verifiers.v1.agent.logger, "warning") as warn:
        _check_borrowed_placement(task, box)
    warn.assert_called_once()
    assert "never re-provisioned" in warn.call_args[0][0]


def test_borrowed_matching_box_is_silent():
    box = make_runtime(DockerConfig(image="python:3.12-slim"))
    task = vf.Task(vf.TaskData(idx=0, prompt="hi", image="python:3.12-slim"))
    with patch.object(verifiers.v1.agent.logger, "warning") as warn:
        _check_borrowed_placement(task, box)
    warn.assert_not_called()


async def test_shared_tools_hit_the_mcp_pairing_guard():
    """`shared_tools` puts MCP in play exactly like a task's own tool servers, so a
    non-MCP harness must be refused — the same `validate_pairing` check an eval runs
    at init against the taskset's declared tools. Raises before any runtime or model
    I/O, so a stub server value is safe."""

    class NoMcpHarness(DefaultHarness):
        SUPPORTS_MCP = False

    ctx = vf.ModelContext(model="test-model", client=None)  # type: ignore[arg-type]
    agent = vf.Agent(
        NoMcpHarness(DefaultHarnessConfig()),
        ctx,
        shared_tools={"search": object()},  # type: ignore[dict-item]
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    with pytest.raises(ValueError, match="does not support MCP"):
        await agent.run(task)


# --- the user channel: guards + ChatSession mechanics (no live calls) ------------


class NoUserSimHarness(DefaultHarness):
    SUPPORTS_USER_SIM = False


def _no_user_agent() -> vf.Agent:
    ctx = vf.ModelContext(model="test-model", client=None)  # type: ignore[arg-type]
    return vf.Agent(NoUserSimHarness(DefaultHarnessConfig()), ctx)


async def _noop_user(message: str) -> vf.Messages:
    return []


async def test_user_requires_supporting_harness():
    """`user=` needs a harness whose program consumes injected user turns; the guard
    fires per run (the channel is run-scoped, not task-scoped) before any I/O."""
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    with pytest.raises(ValueError, match="cannot host a user"):
        await _no_user_agent().run(task, user=_noop_user)


async def test_chat_requires_supporting_harness():
    task = vf.Task(vf.TaskData(idx=0, prompt=None))
    with pytest.raises(ValueError, match="cannot host a user"):
        async with _no_user_agent().chat(task):
            pass


async def test_chat_refuses_prompted_task():
    """chat() opens the conversation itself, so a prompted task is a shape error —
    its first reply would answer the prompt, not the first `turn()`."""
    task = vf.Task(vf.TaskData(idx=0, prompt="already opened"))
    with pytest.raises(ValueError, match="must have no prompt"):
        async with _agent().chat(task):
            pass


def _stub_trace() -> vf.Trace:
    from verifiers.v1.trace import Trace, TraceTask

    return Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)))


async def test_chat_session_turn_roundtrip(monkeypatch):
    """The full session protocol against a stubbed run that speaks the interception's
    user contract: opening ping first (no assistant turn yet), one user call per
    model turn, empty return ends the exchange. Every turn() round-trips, the trace
    is live mid-session, and closing the context stops the run as user_closed."""
    agent = _agent()
    trace = _stub_trace()

    async def fake_run(task, *, runtime=None, user=None, on_trace=None):
        on_trace(trace)
        messages = await user("")  # the opening ping: prompt-less task
        while True:
            messages = await user(f"echo: {messages[0].content}")
            if not messages:
                trace.stop("user_closed")
                return trace

    monkeypatch.setattr(agent, "run", fake_run)
    task = vf.Task(vf.TaskData(idx=0, prompt=None))
    async with agent.chat(task) as session:
        first = await session.turn("one")
        assert (first.text, first.stopped) == ("echo: one", False)
        assert session.trace is trace  # live before close
        second = await session.turn("two")
        assert second.text == "echo: two"
    assert trace.stop_condition == "user_closed"


async def test_chat_close_without_turns(monkeypatch):
    """Leaving the context before any turn() still runs the rollout to a clean end:
    the opening ping resolves empty (the caller closed) and the run finishes."""
    agent = _agent()
    trace = _stub_trace()

    async def fake_run(task, *, runtime=None, user=None, on_trace=None):
        on_trace(trace)
        opening = await user("")
        assert opening == []  # the close resolved the opening: user gone
        trace.stop("user_closed")
        return trace

    monkeypatch.setattr(agent, "run", fake_run)
    async with agent.chat(vf.Task(vf.TaskData(idx=0, prompt=None))):
        pass
    assert trace.stop_condition == "user_closed"


async def test_chat_run_ending_first_stops_the_session(monkeypatch):
    """When the run ends on its own (a limit, a @stop, a harness exit), a caller
    mid-turn() gets the stopped marker instead of hanging, and further turns raise."""
    agent = _agent()
    trace = _stub_trace()

    async def fake_run(task, *, runtime=None, user=None, on_trace=None):
        on_trace(trace)
        await user("")  # consume the first turn's message...
        trace.stop("max_turns")  # ...then end without answering (limit refused)
        return trace

    monkeypatch.setattr(agent, "run", fake_run)
    async with agent.chat(vf.Task(vf.TaskData(idx=0, prompt=None))) as session:
        reply = await session.turn("hello?")
        assert reply.stopped and reply.text == ""
        with pytest.raises(RuntimeError, match="over"):
            await session.turn("still there?")
    assert trace.stop_condition == "max_turns"


async def test_chat_run_end_before_turn_still_returns_stopped(monkeypatch):
    """The run's end has one surface: landing between turns (not mid-turn), the next
    turn() drains the queued stopped marker instead of racing to a RuntimeError — a
    seat that fails fast forfeits deterministically (kuhn-poker's ask())."""
    import asyncio

    agent = _agent()
    trace = _stub_trace()

    async def fake_run(task, *, runtime=None, user=None, on_trace=None):
        on_trace(trace)
        trace.stop("harness_timeout")  # dies before ever consulting the user
        return trace

    monkeypatch.setattr(agent, "run", fake_run)
    async with agent.chat(vf.Task(vf.TaskData(idx=0, prompt=None))) as session:
        while not session._ended:  # let the run land its done-callback first
            await asyncio.sleep(0)
        reply = await session.turn("hello?")
        assert reply.stopped
        with pytest.raises(RuntimeError, match="over"):
            await session.turn("anyone?")
    assert trace.stop_condition == "harness_timeout"


async def test_chat_closed_user_is_idempotent(monkeypatch):
    """A run that consults its user again after consuming the close sentinel gets []
    forever instead of hanging on the drained queue."""
    agent = _agent()
    trace = _stub_trace()

    async def fake_run(task, *, runtime=None, user=None, on_trace=None):
        on_trace(trace)
        assert await user("") == []  # the close resolved the opening...
        assert await user("one more?") == []  # ...and every consult after it
        trace.stop("user_closed")
        return trace

    monkeypatch.setattr(agent, "run", fake_run)
    async with agent.chat(vf.Task(vf.TaskData(idx=0, prompt=None))):
        pass
    assert trace.stop_condition == "user_closed"
