"""Agent guards that fire before any model or runtime I/O (no live calls)."""

from unittest.mock import patch

import pytest

import verifiers.v1 as vf
import verifiers.v1.agent
from verifiers.v1.agent import _check_borrowed_placement
from verifiers.v1.harnesses.default import DefaultHarness, DefaultHarnessConfig
from verifiers.v1.runtimes import DockerConfig, SubprocessConfig, make_runtime


def _agent() -> vf.Agent:
    # The guard under test raises before the model leg is touched, so the
    # client can be a stub.
    return vf.Agent(
        DefaultHarness(DefaultHarnessConfig()),
        "test-model",
        None,  # type: ignore[arg-type]
    )


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

    agent = vf.Agent(
        NoMcpHarness(DefaultHarnessConfig()),
        "test-model",
        None,  # type: ignore[arg-type]
        shared_tools={"search": object()},  # type: ignore[dict-item]
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    with pytest.raises(ValueError, match="does not support MCP"):
        await agent.run(task)


# --- the user channel: guards + ChatSession mechanics (no live calls) ------------


class NoUserSimHarness(DefaultHarness):
    # No Messages prompt and no native resume(): the exchange has no way to advance
    # past the first segment, so a user can't be hosted (the capability is derived,
    # not declared).
    SUPPORTS_MESSAGE_PROMPT = False


def _no_user_agent() -> vf.Agent:
    return vf.Agent(
        NoUserSimHarness(DefaultHarnessConfig()),
        "test-model",
        None,  # type: ignore[arg-type]
    )


async def test_chat_requires_supporting_harness():
    """chat() needs a harness that can resume an exchange (a Messages prompt for the
    default relaunch, or a native resume()); the guard fires before any I/O."""
    task = vf.Task(vf.TaskData(idx=0, prompt=None))
    with pytest.raises(ValueError, match="cannot host a user"):
        async with _no_user_agent().chat(task):
            pass


async def test_chat_mask_needs_a_prompt():
    """mask_prompt hides the task's prompt from the wire; a prompt-less task has
    nothing to hide and is already caller-opened."""
    task = vf.Task(vf.TaskData(idx=0, prompt=None))
    with pytest.raises(ValueError, match="mask_prompt"):
        async with _agent().chat(task, mask_prompt=True):
            pass


def _stub_trace() -> vf.Trace:
    from verifiers.v1.trace import Trace, TraceTask

    return Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)))


class _FakeRun:
    """A `RolloutRun` stand-in speaking the surface `ChatSession` drives: each
    `step()` pops one scripted action (append an answer, fire a stop, ...) and
    reports continuability the way the real run does."""

    def __init__(self, trace: vf.Trace, script) -> None:
        self.trace = trace
        self._script = list(script)
        self.closed = False

    @property
    def ok(self) -> bool:
        return self.trace.stop_condition is None

    async def step(self, messages=None) -> bool:
        if self.closed or not self.ok:
            return False
        turns = self.trace.num_turns
        if self._script:
            self._script.pop(0)(self.trace)
        return self.ok and self.trace.num_turns > turns

    async def close(self) -> vf.Trace:
        self.closed = True
        self.trace.is_completed = True
        return self.trace


def _answer(text: str):
    """A scripted segment that answers: one sampled assistant turn on the trace."""
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage

    def act(trace: vf.Trace) -> None:
        trace.nodes.append(
            MessageNode(message=AssistantMessage(content=text), sampled=True)
        )

    return act


def _session(script) -> vf.ChatSession:
    from verifiers.v1.agent import ChatSession

    return ChatSession(_FakeRun(_stub_trace(), script))  # type: ignore[arg-type]


async def test_chat_session_turn_roundtrip():
    """The session protocol over segments: each turn() runs one segment and returns
    its answer, the trace is live mid-session, and close() stops the exchange as
    user_closed."""
    session = _session([_answer("echo: one"), _answer("echo: two")])
    first = await session.turn("one")
    assert (first.text, first.stopped) == ("echo: one", False)
    assert session.trace.num_turns == 1  # live before close
    second = await session.turn("two")
    assert second.text == "echo: two"
    trace = await session.close()
    assert trace.stop_condition == "user_closed"


async def test_chat_close_without_turns():
    """Closing before any turn() still finishes the rollout cleanly as user_closed."""
    session = _session([])
    trace = await session.close()
    assert trace.stop_condition == "user_closed"
    assert trace.is_completed


async def test_chat_run_ending_first_stops_the_session():
    """When the segment ends the run instead of answering (a limit, a @stop), the
    caller gets the stopped marker, and further turns raise."""
    session = _session([lambda trace: trace.stop("max_turns")])
    reply = await session.turn("hello?")
    assert reply.stopped and reply.text == ""
    with pytest.raises(RuntimeError, match="over"):
        await session.turn("still there?")
    assert session.trace.stop_condition == "max_turns"


async def test_chat_answer_then_stop_delivers_both():
    """A segment that answers AND ends the exchange (a limit fired after the turn
    committed) delivers the answer now and the stopped marker on the next turn —
    an answer is never swallowed by the stop that followed it."""

    def answer_then_stop(trace: vf.Trace) -> None:
        _answer("last words")(trace)
        trace.stop("max_turns")

    session = _session([answer_then_stop])
    reply = await session.turn("hello?")
    assert (reply.text, reply.stopped) == ("last words", False)
    final = await session.turn("more?")
    assert final.stopped


async def test_chat_closed_session_refuses_turns():
    """close() is idempotent, and a turn() after it is a caller bug, not a hang."""
    session = _session([_answer("hi")])
    await session.turn("hello")
    trace = await session.close()
    assert trace is await session.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        await session.turn("anyone?")


async def test_prompted_chat_opens_with_a_bare_turn():
    """A prompted task speaks first: turn(message) before the opening reply is a
    shape error, and a bare turn() on an already-open exchange is too."""
    from verifiers.v1.agent import ChatSession
    from verifiers.v1.trace import Trace, TraceTask

    trace = Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="the board"))
    )
    session = ChatSession(_FakeRun(trace, [_answer("first move")]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bare turn()"):
        await session.turn("eager reply")
    first = await session.turn()
    assert first.text == "first move"
    with pytest.raises(ValueError, match="nothing to run"):
        await session.turn()
