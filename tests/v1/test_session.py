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


def make_session(task: vf.Task) -> Session:
    session = Session()
    session._rollout = SimpleNamespace(trace=vf.Trace(task=task))
    return session


def null_agent(**kwargs) -> vf.Agent:
    return vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(model="org/model", client=object()),
        **kwargs,
    )


async def test_session_handshake_and_end():
    """The lockstep: opening `_respond("")` delivers nothing and returns the first
    turn; each later `_respond(reply)` delivers the reply and suspends; `end()`
    unblocks the suspended episode into a clean exit and stops the trace."""
    session = make_session(vf.Task(idx=0, prompt=None))
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
    session = make_session(vf.Task(idx=0, prompt=None))

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
    session = make_session(vf.Task(idx=0, prompt=None))
    session._episode = asyncio.create_task(asyncio.sleep(30))
    try:
        first = asyncio.create_task(session.turn("a"))
        await asyncio.sleep(0)  # let it post and suspend
        with pytest.raises(RuntimeError, match="one driver per seat"):
            await session.turn("b")
        first.cancel()
    finally:
        session._episode.cancel()


async def test_interact_refusals():
    """The refuse-loudly table: a prompted task, a task with its own user simulator,
    and a harness that can't take injected user turns are all rejected at the call."""
    agent = null_agent()
    with pytest.raises(ValueError, match="opened by the first turn"):
        async with agent.interact(vf.Task(idx=0, prompt="hi")):
            pass

    class SimTask(vf.Task):
        def load_user(self):  # pragma: no cover - never called
            return object()

    with pytest.raises(ValueError, match="second claimant"):
        async with agent.interact(SimTask(idx=0, prompt=None)):
            pass

    class MuteHarnessConfig(HarnessConfig):
        pass

    class MuteHarness(Harness[MuteHarnessConfig]):
        SUPPORTS_USER_SIM = False

        async def launch(self, *a, **k):  # pragma: no cover - never called
            raise NotImplementedError

    mute = vf.Agent(
        MuteHarness(MuteHarnessConfig()),
        vf.ModelContext(model="org/model", client=object()),
    )
    with pytest.raises(ValueError, match="cannot take injected user turns"):
        async with mute.interact(vf.Task(idx=0, prompt=None)):
            pass


async def test_interact_lifecycle_with_stubbed_rollout(monkeypatch):
    """`Agent.interact` end to end minus the model: the stubbed `Rollout.run` consumes
    the programmatic user seat exactly like the interception loop, and the scope exit
    ends the episode, joins it, and stamps provenance + parents."""

    async def fake_run(self):
        self.trace = trace = vf.Trace(task=self.task)
        msgs = await self._user("")  # opening
        while msgs:
            msgs = await self._user(f"echo:{msgs[0].content}")
        if trace.stop_condition is None:  # pragma: no cover - end() always set it
            trace.stop("agent_completed")
        return trace

    monkeypatch.setattr(Rollout, "run", fake_run)
    agent = null_agent(name="white", trainable=False)
    parent = vf.Trace(task=vf.Task(idx=0, prompt="seed"))

    async with agent.interact(vf.Task(idx=1, prompt=None), parents=[parent]) as session:
        assert await session.turn("kick off") == "echo:kick off"
        trace = session.trace  # live handle
        trace.info["game"] = {"score": 1.0}

    assert trace.stop_condition == "interaction_complete"
    assert trace.agent == "white" and trace.trainable is False
    assert trace.parents == [parent.id]
    assert trace.info["game"] == {"score": 1.0}
    assert trace.info["agent"]["name"] == "white"
