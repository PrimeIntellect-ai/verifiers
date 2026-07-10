"""Session tests: the queue handshake, the safety contract, and `Agent.interact`.

The handshake tests drive `Session._respond` from a fake interception loop, so the
turn/reply lockstep, `end()`, and the poison paths run without a model or a rollout.
The lifecycle test monkeypatches `Rollout.run` with a stub that consumes the session's
`_respond` exactly as the interception loop does — so `Agent.interact`'s background
task, scope-exit stop, and provenance stamp run for real."""

import asyncio
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.agent import Session
from verifiers.v1.trace import TraceTask


def task_of(prompt) -> vf.Task:
    return vf.Task(vf.TaskData(idx=0, prompt=prompt))


def trace_for(task: vf.Task) -> vf.Trace:
    return vf.Trace(task=TraceTask(type=type(task).__name__, data=task.data))


def make_session(task: vf.Task) -> Session:
    session = Session()
    session._rollout = SimpleNamespace(trace=trace_for(task))
    return session


def null_agent(**kwargs) -> vf.Agent:
    return vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(
            model="org/model", client=object(), sampling=vf.SamplingConfig()
        ),
        **kwargs,
    )


async def test_session_handshake_and_end():
    """The lockstep: opening `_respond("")` delivers nothing and returns the first
    turn; each later `_respond(reply)` delivers the reply and suspends; `end()`
    unblocks the suspended episode into a clean exit and stops the trace."""
    session = make_session(task_of(None))
    seen: list[str] = []

    async def fake_interception_loop():
        msgs = await session._respond("")  # the opening call
        while msgs:
            seen.append(msgs[0].content)
            msgs = await session._respond(f"echo:{msgs[0].content}")

    episode = asyncio.create_task(fake_interception_loop())
    session._episode = episode
    episode.add_done_callback(session._poison)

    assert await session.turn("one") == "echo:one"
    assert await session.turn("two") == "echo:two"
    await session.end()
    await episode
    assert seen == ["one", "two"]
    assert session.trace.stop_condition == "interaction_complete"


async def test_turn_raises_on_dead_episode():
    """`turn()` never hangs on an episode that can't answer: a seat that died
    mid-await gets poisoned, and a later `turn()` refuses up front."""
    session = make_session(task_of(None))

    async def dying_episode():
        await session._respond("")  # take the first turn, then die before replying
        raise RuntimeError("boom")

    episode = asyncio.create_task(dying_episode())
    session._episode = episode
    episode.add_done_callback(session._poison)

    with pytest.raises(vf.SessionEnded):
        await session.turn("hello?")  # mid-await death → poisoned reply
    with pytest.raises(vf.SessionEnded):
        await session.turn("still there?")  # already-dead → refused up front
    with pytest.raises(RuntimeError, match="boom"):
        await episode


async def test_double_drive_refused():
    """One driver per seat: a second `turn()` while one is pending raises instead of
    interleaving two conversations."""
    session = make_session(task_of(None))
    session._episode = asyncio.create_task(asyncio.sleep(30))
    try:
        first = asyncio.create_task(session.turn("a"))
        await asyncio.sleep(0)  # let it post and suspend
        with pytest.raises(RuntimeError, match="one driver per seat"):
            await session.turn("b")
        first.cancel()
    finally:
        session._episode.cancel()


async def test_cancelled_turn_desyncs_loudly():
    """A `turn()` cancelled mid-flight (e.g. a sibling seat in a `gather` raised) left
    its message with the model — the conversation is desynced, so a later `turn()`
    refuses instead of consuming the stale reply one turn late."""
    session = make_session(task_of(None))
    session._episode = asyncio.create_task(asyncio.sleep(30))
    try:
        pending = asyncio.create_task(session.turn("in flight"))
        await asyncio.sleep(0)  # let it post its message and suspend
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(RuntimeError, match="cancelled mid-flight"):
            await session.turn("resync?")
        await session.end()  # the sanctioned way out of a desynced seat
    finally:
        session._episode.cancel()


def test_extend_coerces_reasoning_only_turn():
    """Regression (found by a live chess game): a truncated reasoning turn comes back
    with `content: null` and no tool calls — valid *output*, but upstreams 422 it when
    re-sent as conversation *input*. `extend` must coerce it to the empty string; a
    tool-call turn stays legitimately content-less."""
    from verifiers.v1.dialects.chat import ChatDialect

    dialect = ChatDialect()
    body = {"messages": [{"role": "user", "content": "your move"}]}
    truncated = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    extended = dialect.extend(body, truncated, [vf.UserMessage(content="next")])
    assert extended["messages"][1] == {"role": "assistant", "content": ""}

    tool_turn = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "x"}],
                }
            }
        ]
    }
    extended = dialect.extend(body, tool_turn, [])
    assert extended["messages"][1]["content"] is None  # untouched — tool calls carry it


async def test_interact_refusals():
    """The refuse-loudly table: a prompted task, a task with its own user simulator,
    and a harness that can't take injected user turns are all rejected at the call."""
    agent = null_agent()
    with pytest.raises(ValueError, match="opened by the first turn"):
        async with agent.interact(task_of("hi")):
            pass

    class SimTask(vf.Task):
        user = vf.User  # declared simulator class — never instantiated here

    with pytest.raises(ValueError, match="second claimant"):
        async with agent.interact(SimTask(vf.TaskData(idx=0, prompt=None))):
            pass

    class MuteHarnessConfig(HarnessConfig):
        pass

    class MuteHarness(Harness[MuteHarnessConfig]):
        SUPPORTS_USER_SIM = False

        async def launch(self, *a, **k):  # pragma: no cover - never called
            raise NotImplementedError

    mute = vf.Agent(
        MuteHarness(MuteHarnessConfig()),
        vf.ModelContext(
            model="org/model", client=object(), sampling=vf.SamplingConfig()
        ),
    )
    with pytest.raises(ValueError, match="cannot take injected user turns"):
        async with mute.interact(task_of(None)):
            pass


async def test_interact_lifecycle_with_stubbed_rollout(monkeypatch):
    """`Agent.interact` end to end minus the model: the stubbed `Rollout.run` consumes
    the programmatic user seat exactly like the interception loop, and the scope exit
    ends the episode, joins it, and stamps provenance + parents."""

    async def fake_run(self):
        self.trace = trace = vf.Trace(
            task=TraceTask(type=type(self.task).__name__, data=self.task.data)
        )
        msgs = await self._user("")  # opening
        while msgs:
            msgs = await self._user(f"echo:{msgs[0].content}")
        if trace.stop_condition is None:  # pragma: no cover - end() always set it
            trace.stop("agent_completed")
        return trace

    monkeypatch.setattr(Rollout, "run", fake_run)
    agent = null_agent(name="white", trainable=False)
    parent = trace_for(task_of("seed"))

    async with agent.interact(
        vf.Task(vf.TaskData(idx=1, prompt=None)), parents=[parent]
    ) as session:
        assert await session.turn("kick off") == "echo:kick off"
        trace = session.trace  # live handle
        trace.info["game"] = {"score": 1.0}

    assert trace.stop_condition == "interaction_complete"
    assert trace.agent == "white" and trace.trainable is False
    assert trace.parents == [parent.id]
    assert trace.info["game"] == {"score": 1.0}
    assert trace.info["agent"]["name"] == "white"
