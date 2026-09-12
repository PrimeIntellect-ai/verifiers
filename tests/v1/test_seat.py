"""The seat: one turn on a leased box (prepared once per box, parked under its lease), a
session kept open across turns with the one nudge, the fault rule (a permanent refusal
ends the run, a transient death is a `Fault` unless it is the agent's evidence),
`during` and `serve` around the interaction, the seat's record and live trace; the
trace-error classifiers. A scripted agent over real subprocess runtimes, no model."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1 import seat as seats
from verifiers.v1.agent import Segment
from verifiers.v1.errors import HarnessError, ProviderError, SandboxError
from verifiers.v1.runtimes import (
    RuntimePool,
    RuntimePoolConfig,
    SubprocessConfig,
    provision_runtime,
)
from verifiers.v1.runtimes.durable import Box, InfraError
from verifiers.v1.seat import (
    Fault,
    LiveTrace,
    ModelConfigurationError,
    Opening,
    Seat,
    Session,
)
from verifiers.v1.types import AssistantMessage, UserMessage

DIED = SimpleNamespace(
    type="HarnessError", message="harness 'kernel': exited 1", status_code=None
)
PROVIDER = SimpleNamespace(
    type="ProviderError", message="upstream unavailable", status_code=503
)
REFUSED = SimpleNamespace(type="ProviderError", message="forbidden", status_code=403)
EXPIRED = SimpleNamespace(
    type="HarnessError",
    message="agent timeout: rollout exceeded its 60s budget",
    status_code=None,
)


def trace_error(message, kind="ProviderError", status=None):
    return SimpleNamespace(type=kind, message=message, status_code=status)


class Scripted:
    """An `Agent` in miniature: `provision` a subprocess runtime (the pool's under
    `reuse`), `interaction` a scripted exchange whose every `turn` takes the next
    `turns` entry (a str: the reply; a trace error: the death; a callable `fn(message)`:
    awaited, its return the reply); `fails` is raised when the interaction opens."""

    def __init__(self, runtimes: RuntimePool | None = None):
        self.runtimes = runtimes
        self.config = SimpleNamespace(
            model="m", client=SimpleNamespace(base_url="http://provider.invalid/v1")
        )
        self.turns: list = []
        self.heard: list = []
        self.opened: list = []
        self.fails: BaseException | None = None
        self.live = 0

    @asynccontextmanager
    async def provision(self, task, *, reuse=None):
        if reuse is not None and self.runtimes is not None:
            async with self.runtimes.lease(reuse, SubprocessConfig(), {}) as runtime:
                yield runtime
            return
        async with provision_runtime(SubprocessConfig()) as runtime:
            yield runtime

    @asynccontextmanager
    async def interaction(self, task, *, runtime, on_trace):
        if self.fails is not None:
            raise self.fails
        trace = SimpleNamespace(
            id=f"t{len(self.opened) + 1}",
            nodes=[],
            calls=[],
            last_error=None,
            stop_condition=None,
        )
        trace.to_record = lambda: {"nodes": len(trace.nodes), "calls": len(trace.calls)}
        on_trace(trace)
        self.opened.append(task)
        self.live += 1

        async def turn(message=None):
            self.heard.append(message)
            action = self.turns.pop(0)
            if callable(action):
                action = await action(message)
            if isinstance(action, str):
                trace.nodes.append(
                    SimpleNamespace(message=UserMessage(content=message or ""))
                )
                reply = AssistantMessage(content=action)
                trace.nodes.append(SimpleNamespace(message=reply))
                trace.calls.append({})
                return Segment(messages=[reply])
            trace.last_error, trace.stop_condition = action, "error"
            return Segment(messages=[], terminated=True)

        try:
            yield SimpleNamespace(trace=trace, turn=turn)
        finally:
            self.live -= 1


def opening(**fields) -> Opening:
    task = vf.Task(
        vf.TaskData(prompt=[UserMessage(content="go")], system_prompt="be brief"),
        vf.TaskConfig(),
    )
    return Opening(task=task, key="k", name="k.trace", **fields)


async def test_a_turn_leases_a_box_prepares_it_once_and_hands_it_to_the_harvest():
    """A leased box is prepared when this seat never prepared it, reused as it stands
    after; without a lease the box is fresh every turn and closed after; `discard` stops
    the parked box."""
    async with RuntimePool(RuntimePoolConfig(ttl=60)) as runtimes:
        agent = Scripted(runtimes)
        lines: list[str] = []
        seat = Seat("miner", agent, log=lines.append)
        prepared: list[bool] = []

        async def prepare(box: Box, fresh: bool) -> None:
            prepared.append(fresh)
            if fresh:
                await box.run("echo prepared > marker")

        async def harvest(session: Session) -> str:
            assert session.segment is not None and session.segment.last_reply == "done"
            return f"{session.box.id}:{await session.box.read(f'{session.box.workdir}/marker')}"

        agent.turns = ["done", "done", "done", "done"]
        first = await seat.turn(opening(lease="miner:acme", prepare=prepare), harvest)
        second = await seat.turn(opening(lease="miner:acme", prepare=prepare), harvest)
        assert (
            first == second
            and first.endswith(":prepared\n")
            and prepared == [True, False]
        )
        assert agent.heard == [None, None]  # a prompted task: the opening turn is bare
        assert [line.rsplit(" ", 1)[1] for line in lines] == ["up", "reused"]
        await seat.discard("miner:acme")
        assert (
            await seat.turn(opening(lease="miner:acme", prepare=prepare), harvest)
            != first
        )
        assert prepared == [True, False, True]
        box_id = await seat.turn(opening(prepare=prepare), lambda session: _id(session))
        assert box_id not in seat.prepared and prepared[-1] is True
        assert agent.live == 0


async def _id(session: Session) -> str:
    return session.box.id


async def test_a_session_keeps_one_interaction_across_turns_and_the_nudge_fires_once_when_the_box_stands_unchanged():
    """The first turn is the prompt's (bare), every later one the message given;
    `unchanged(box)` asked after a reply without a call sends the nudge as one more user
    turn, once per turn; a terminated segment or a death asks nothing."""
    agent = Scripted()
    seat = Seat("coordinator", agent)
    asked: list[str] = []

    async def unchanged(box: Box) -> bool:
        asked.append(box.id)
        return len(asked) == 1

    agent.turns = ["a reply, no call", "acted", "acted again", "and again"]
    async with seat.session(
        opening(unchanged=unchanged, nudge="[harness] nothing changed; act")
    ) as session:
        seg = await session.turn()
        assert seg.last_reply == "acted" and agent.heard == [
            None,
            "[harness] nothing changed; act",
        ]
        seg = await session.turn("more")
        assert (
            seg.last_reply == "acted again"
            and agent.heard[-1] == "more"
            and len(asked) == 2
        )
        assert (
            session.turns == 2 and session.died is None and len(session.messages) == 6
        )
        assert agent.live == 1 and len(agent.opened) == 1
    assert agent.live == 0


async def test_a_transient_death_is_a_fault_unless_it_is_the_agents_evidence_and_a_refusal_ends_the_run():
    agent = Scripted()
    seat = Seat("solver", agent)
    agent.turns = [PROVIDER]
    with pytest.raises(Fault) as fault:
        async with seat.session(opening()) as session:
            await session.turn()
    assert (
        fault.value.detail == "ProviderError: upstream unavailable"
        and fault.value.fresh is False
    )
    agent.turns = [PROVIDER]
    async with seat.session(opening(evidence=True)) as session:
        seg = await session.turn()
        assert seg.terminated and session.died == "ProviderError: upstream unavailable"
    agent.turns = [EXPIRED]
    async with seat.session(
        opening()
    ) as session:  # the spent budget is the caller's to read, never a fault
        await session.turn()
        assert session.died == EXPIRED.type + ": " + EXPIRED.message
    agent.turns = [REFUSED]
    with pytest.raises(
        ModelConfigurationError,
        match="solver .m at http://provider.invalid/v1.: permanent",
    ) as refused:
        await seat.turn(opening(evidence=True), _id)
    assert refused.value.status_code == 403
    agent.turns = [DIED]
    with pytest.raises(Fault, match="harness 'kernel': exited 1"):
        await seat.turn(opening(), _id)


async def test_what_fails_around_the_interaction_is_a_fault_naming_whether_the_box_is_gone():
    agent = Scripted()
    seat = Seat("judge", agent)
    agent.fails = HarnessError("kernel did not start")
    with pytest.raises(Fault) as harness:
        await seat.turn(opening(), _id)
    assert (
        harness.value.fresh is False
        and "could not open: kernel did not start" in harness.value.detail
    )
    agent.fails = SandboxError("exec failed")
    with pytest.raises(Fault) as sandbox:
        await seat.turn(opening(), _id)
    assert (
        sandbox.value.fresh is True
        and "could not open on the box" in sandbox.value.detail
    )
    agent.fails = None

    async def infra(box, fresh):
        raise InfraError("read /x failed: gone")

    async def defect(box, fresh):
        raise KeyError("oops")

    with pytest.raises(Fault, match="read /x failed: gone") as fault:
        await seat.turn(opening(prepare=infra), _id)
    assert fault.value.fresh is True
    with pytest.raises(Fault, match="KeyError: 'oops'"):
        async with seat.session(opening(prepare=defect)):
            pass
    assert agent.live == 0 and "oops" not in seat.prepared


async def test_during_runs_beside_the_interaction_and_serve_wraps_it():
    agent = Scripted()
    seat = Seat("builder", agent)
    events: list[str] = []

    async def during(box: Box) -> None:
        events.append("beside")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("beside ended")

    @asynccontextmanager
    async def serve(it):
        events.append(f"serving {it.trace.id}")
        yield
        events.append("served")

    async def reply(message):
        await asyncio.sleep(0.01)
        return "ok"

    async def harvest(session: Session) -> None:
        events.append("harvest")

    agent.turns = [reply]
    await seat.turn(opening(during=during, serve=serve), harvest)
    assert events == [
        "serving t1",
        "beside",
        "beside ended",
        "served",
        "harvest",
    ]  # the harvest reads a quiet box


async def test_the_seat_records_the_trace_and_the_live_trace_is_written_while_it_runs_and_removed_after(
    tmp_path,
):
    class Recording(Seat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorded: list = []

        def live(self, session: Session) -> LiveTrace:
            return LiveTrace(
                session.traces, tmp_path / f"{session.name}.live.json", period=0.01
            )

        def record(self, session: Session) -> None:
            self.recorded.append((session.name, len(session.traces), session.died))

    agent = Scripted()
    seat = Recording("reviewer", agent)
    seen: list[dict] = []

    async def slow(message):
        for _ in range(50):
            await asyncio.sleep(0.005)
            if (tmp_path / "k.trace.live.json").exists():
                seen.append(json.loads((tmp_path / "k.trace.live.json").read_text()))
                break
        return "done"

    agent.turns = [slow]
    await seat.turn(opening(), _id)
    assert seen and seen[0] == {"nodes": 0, "calls": 0}
    assert not (tmp_path / "k.trace.live.json").exists() and seat.recorded == [
        ("k.trace", 1, None)
    ]
    agent.fails = HarnessError("no")
    with pytest.raises(Fault):
        await seat.turn(opening(), _id)
    assert seat.recorded[-1] == (
        "k.trace",
        0,
        None,
    )  # never opened: no trace, still recorded


def test_a_live_trace_is_written_only_when_the_trace_changed_and_a_failing_writer_is_logged(
    tmp_path, caplog
):
    trace = SimpleNamespace(nodes=[], calls=[], stop_condition=None)
    trace.to_record = lambda: {"calls": list(trace.calls)}
    live = LiveTrace([trace], tmp_path / "t.live.json", period=600)
    live.write()
    first = live.path.stat().st_mtime_ns
    live.write()
    assert live.path.stat().st_mtime_ns == first
    trace.calls.append({"usage": {}})
    live.write()
    assert json.loads(live.path.read_text())["calls"] == [{"usage": {}}]
    live.remove()
    live.remove()  # already gone: nothing
    assert not live.path.exists()

    def failing(traces, path: Path) -> None:
        raise OSError("disk full")

    broken = LiveTrace([trace], tmp_path / "b.live.json", period=600, write=failing)
    broken.write()
    assert (
        "live trace" in caplog.text
        and "disk full" in caplog.text
        and not broken.path.exists()
    )
    LiveTrace([], tmp_path / "none.json", period=600).write()
    assert not (tmp_path / "none.json").exists()


async def test_a_live_trace_ticks_on_a_timer_and_zero_arms_nothing(tmp_path):
    trace = SimpleNamespace(nodes=[], calls=[], stop_condition=None)
    trace.to_record = lambda: {"calls": len(trace.calls)}
    live = LiveTrace([trace], tmp_path / "t.live.json", period=0.01)
    live.start()
    await asyncio.sleep(0.05)
    assert live.path.exists()
    live.stop()
    trace.calls.append({})
    await asyncio.sleep(0.05)
    assert json.loads(live.path.read_text()) == {"calls": 0}  # stopped: no more ticks
    off = LiveTrace([trace], tmp_path / "off.json", period=0)
    off.start()
    await asyncio.sleep(0.02)
    assert off._timer is None and not off.path.exists()


# --- the trace-error classifiers


@pytest.mark.parametrize(
    ("error", "permanent"),
    [
        (trace_error("invalid api key", status=401), True),
        (trace_error("forbidden", status=403), True),
        (ProviderError("bad key", status_code=401), True),
        (ModelConfigurationError("earlier refusal", status_code=403), True),
        # every other status is a bounded retry: the code, not the prose, decides
        (trace_error("model not found", status=404), False),
        (trace_error("unknown parameter", status=400), False),
        (trace_error("rate limited", status=429), False),
        (trace_error("overloaded", status=503), False),
        (ProviderError("bad gateway", status_code=502), False),
        # a status only in the prose is no status at all
        (trace_error("Error code: 401 - invalid api key"), False),
        # not a provider error
        (
            trace_error(
                "agent timeout: rollout exceeded its 60s budget", kind="HarnessError"
            ),
            False,
        ),
        (SandboxError("status 401: placement failed"), False),
    ],
)
def test_only_a_typed_401_or_403_is_a_permanent_provider_error(error, permanent):
    assert seats.permanent_provider_error(error) is permanent


def test_provider_status_reads_only_the_typed_code():
    assert (
        seats.provider_status(trace_error("Error code: 503 - upstream 502", status=429))
        == 429
    )
    assert seats.provider_status(trace_error("Error code: 503 - upstream 502")) is None
    assert seats.provider_status(ProviderError("x", status_code=418)) == 418
    assert seats.provider_status(SandboxError("status 500")) is None
    assert seats.provider_status(None) is None


@pytest.mark.parametrize(
    ("error", "fault"),
    [
        (trace_error("overloaded", status=503), True),
        (trace_error("model not found", status=404), True),
        (trace_error("harness 'kernel': exited 1", kind="HarnessError"), True),
        (trace_error("forbidden", status=403), False),
        (
            trace_error(
                "agent timeout: rollout exceeded its 60s budget", kind="HarnessError"
            ),
            False,
        ),
        (trace_error("exec failed", kind="SandboxError"), False),
        (None, False),
    ],
)
def test_a_provider_fault_is_a_transient_provider_or_harness_death(error, fault):
    assert seats.provider_fault(error) is fault


def test_the_spent_rollout_budget_is_the_harness_timeout_alone():
    assert seats.rollout_budget_expired(
        trace_error(
            "agent timeout: rollout exceeded its 60s budget", kind="HarnessError"
        )
    )
    assert not seats.rollout_budget_expired(
        trace_error("harness 'kernel': agent process exited", kind="HarnessError")
    )
    assert not seats.rollout_budget_expired(
        trace_error("agent timeout", kind="ProviderError")
    )
    assert not seats.rollout_budget_expired(None)


def test_configuration_error_names_the_seat_model_endpoint_and_provider_error_without_credentials():
    config = SimpleNamespace(
        model="fake/model",
        client=SimpleNamespace(
            base_url="https://user:secret@provider.invalid:8443/v1?key=abc"
        ),
    )
    error = seats.configuration_error(
        "solver", config, trace_error("invalid api key", status=401)
    )
    assert (
        isinstance(error, ModelConfigurationError)
        and error.retryable is False
        and error.status_code == 401
    )
    assert str(error) == (
        "solver (fake/model at https://provider.invalid:8443/v1): permanent model request failure: "
        "ProviderError: invalid api key"
    )
    assert seats.permanent_provider_error(error)
    bare = seats.configuration_error(
        "judge", None, ProviderError("refused", status_code=403)
    )
    assert (
        str(bare) == "judge: permanent model request failure: ProviderError: refused"
        and bare.status_code == 403
    )
    assert seats.public_url("not a url") == "not a url"


def test_describe_names_a_trace_error_and_an_exception_alike():
    assert (
        seats.describe(trace_error("boom", kind="HarnessError")) == "HarnessError: boom"
    )
    assert seats.describe(HarnessError("exited 1")) == "HarnessError: exited 1"
