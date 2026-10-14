"""Durable calls (`verifiers.v1.durable`): with a durable installed, `agent.run` and
`agent.interaction` find their latest earlier trace (by seat and position, or by
`key=`) through a `TraceLookup` and return the finished one, continue a cut-off one
from its latest checkpoint, or run fresh; `Env.durable(task)` installs one for an
episode, scoped by its trace labels.

Deterministic, no API key: the scripted model, in-process harness and fake
checkpointing boxes of `test_agent_resume`.
"""

# ruff: noqa: F811 - the fixtures imported from test_agent_resume are parameters

import asyncio
import json
from pathlib import Path
from typing import ClassVar

import pytest
from test_agent_resume import (  # noqa: F401 - fixtures
    EVENTS,
    FakeRuntime,
    NotesTask,
    RestoreConfig,
    ScriptHarness,
    box_lost,
    fake_boxes,
    make,
    model,
    roles,
    task,
)

import test_agent_resume as resume_tests
import verifiers.v1 as vf
import verifiers.v1.box as box_module
import verifiers.v1.runtimes as runtimes_module
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.durable import call_parts, call_ref, latest, refs_of, resolve_call
from verifiers.v1.utils.retries import RETRY_NOTE


@pytest.fixture
def store(tmp_path) -> vf.TraceDir:
    return vf.TraceDir(tmp_path / "traces")


class Partial:
    """A lookup over a store someone else writes: here, one trace as saved mid-run."""

    def __init__(self, *records: dict) -> None:
        self.records = list(records)

    def get(self, ref):
        traces = [vf.WireTrace.model_validate(r) for r in self.records]
        return latest([t for t in traces if ref in refs_of(t.labels)])


# -- run(key=) ------------------------------------------------------------------


async def test_a_finished_call_returns_its_trace_without_running(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store, match={"job": "j1"})):
        first = await agent.run(task(), key="solve")
        assert first.ok, first.errors
        assert not first.kept
        assert len(model.requests) == 3
        again = await agent.run(task(), key="solve")
    assert again.kept
    assert again.id == first.id
    assert len(model.requests) == 3
    label = first.labels["call"]
    assert first.labels["job"] == "j1"
    assert label["key"] == "agent/solve"
    assert set(label["parts"]) >= {"task", "task_code", "data", "model", "harness"}
    # Another key, or another scope, runs.
    with vf.use_durable(vf.Durable(store, match={"job": "j2"})):
        other = await agent.run(task(), key="solve")
    assert not other.kept
    assert len(model.requests) == 6


async def test_without_a_durable_installed_nothing_changes(model, store):
    agent = make(model)
    first = await agent.run(task())
    second = await agent.run(task())
    assert first.info == {} and second.info == {}
    assert not first.kept and not second.kept
    assert len(model.requests) == 6
    # depends= is inert on a plain call.
    assert (await agent.run(task(), depends="x")).ok


async def test_every_call_is_durable_by_its_position(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        first = await agent.run(task("one"))
        second = await agent.run(task("two"))
    assert first.labels["call"]["key"] == "agent/0"
    assert second.labels["call"]["key"] == "agent/1"
    assert len(model.requests) == 6
    # Run again: the same calls in the same order find their traces.
    with vf.use_durable(vf.Durable(store)):
        again = [await agent.run(task("one")), await agent.run(task("two"))]
    assert [t.kept for t in again] == [True, True]
    assert [t.id for t in again] == [first.id, second.id]
    assert len(model.requests) == 6


async def test_outside_an_episode_agents_share_one_count(model, store):
    """As docs/v1/agent.md says: every agent outside an episode is named "agent", so
    two agents' calls take positions from one count."""
    solver, reviewer = make(model), make(model)
    with vf.use_durable(vf.Durable(store)):
        first = await solver.run(task("one"))
        second = await reviewer.run(task("two"))
    keys = [t.labels["call"]["key"] for t in (first, second)]
    assert keys == ["agent/0", "agent/1"]


async def test_positions_are_taken_when_calls_start(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        traces = await asyncio.gather(*(agent.run(task(p)) for p in ("a", "b", "c")))
    keys = [t.labels["call"]["key"] for t in traces]
    assert keys == ["agent/0", "agent/1", "agent/2"]
    with vf.use_durable(vf.Durable(store)):
        again = await asyncio.gather(*(agent.run(task(p)) for p in ("a", "b", "c")))
    assert [t.id for t in again] == [t.id for t in traces]
    assert all(t.kept for t in again)


async def test_shifted_positions_run_fresh(model, store):
    """Code that inserts a call shifts every later position onto a trace with another
    fingerprint: those calls run again, none returns a stale result."""
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        await agent.run(task("a"))
        await agent.run(task("b"))
    model.requests.clear()
    with vf.use_durable(vf.Durable(store)):
        shifted = [await agent.run(task(p)) for p in ("new", "a", "b")]
    assert not any(t.kept for t in shifted)
    assert [t.task.data.prompt for t in shifted] == ["new", "a", "b"]
    assert len(model.requests) == 9
    with vf.use_durable(vf.Durable(store)):
        again = [await agent.run(task(p)) for p in ("new", "a", "b")]
    assert all(t.kept for t in again)


async def test_a_key_overrides_the_position_and_durable_false_opts_out(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        named = await agent.run(task(), key="named")
        positioned = await agent.run(task())
        plain = await agent.run(task(), durable=False)
    assert named.labels["call"]["key"] == "agent/named"
    # A named call takes no position.
    assert positioned.labels["call"]["key"] == "agent/0"
    assert plain.info == {}
    with vf.use_durable(vf.Durable(store)):
        assert (await agent.run(task(), key="named")).kept
        assert not (await agent.run(task(), durable=False)).kept
    assert len(model.requests) == 12


async def test_a_key_is_the_seats_and_never_a_position(model, store):
    with vf.use_durable(vf.Durable(store)), pytest.raises(ValueError, match="position"):
        await make(model).run(task(), key="0")


def test_episode_random_is_seeded_by_the_scope(store):
    def draws(match):
        with vf.use_durable(vf.Durable(store, match=match)):
            return [vf.episode_random().random(), vf.episode_random("b").random()]

    assert draws({"job": "j1"}) == draws({"job": "j1"})
    assert draws({"job": "j1"}) != draws({"job": "j2"})
    one = draws({"job": "j1"})
    assert one[0] != one[1]
    # Without a durable nothing reruns: an unseeded generator.
    assert vf.episode_random().random() != vf.episode_random().random()


async def test_a_durable_call_manages_its_own_resume(model, store):
    with pytest.raises(TypeError, match="resumes itself"):
        await make(model).run(task(), key="solve", note="hi")


async def test_budgets_keep_a_call_and_its_inputs_rerun_it(model, store):
    durable = vf.Durable(store)
    with vf.use_durable(durable):
        await make(model).run(task(), key="solve")
        # A raised timeout, more turns, retries: the same call.
        kept = await make(
            model,
            timeout={"rollout": 999},
            max_turns=50,
            retries={"max_retries": 2},
        ).run(task(), key="solve")
        assert kept.kept
        assert len(model.requests) == 3
        # A changed prompt, model or harness runs again.
        assert not (await make(model).run(task("other"), key="solve")).kept
        other = make(model)
        other.config = other.config.model_copy(update={"model": "scripted-2"})
        assert not (await other.run(task(), key="solve")).kept
        harness = vf.Agent(
            vf.AgentConfig(
                model="scripted",
                client={"type": "eval", "base_url": model.url, "api_key_var": "K"},
                runtime={"type": "subprocess"},
                harness={"id": "null", "env": {"MODE": "strict"}},
            )
        )
        parts = call_parts(task(), harness, durable)
        assert resolve_call(durable, "agent/solve", parts).kept is None


async def test_depends_and_durable_parts(model, store, tmp_path):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("v1")
    agent = make(model)
    with vf.use_durable(vf.Durable(store, parts={"version": "1"})):
        await agent.run(task(), key="solve", depends=prompt_file)
        assert (await agent.run(task(), key="solve", depends=prompt_file)).kept
        prompt_file.write_text("v2")
        assert not (await agent.run(task(), key="solve", depends=prompt_file)).kept
    # The durable's parts are in every fingerprint.
    with vf.use_durable(vf.Durable(store, parts={"version": "2"})):
        assert not (await agent.run(task(), key="solve", depends=prompt_file)).kept


async def test_a_failed_call_continues_from_its_latest_checkpoint(model, store):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        failed = await agent.run(task(), key="solve")
        assert not failed.ok
        assert [c.id for c in failed.checkpoints] == ["ckpt-1", "ckpt-2"]
        model.requests.clear()
        EVENTS.clear()
        resumed = await agent.run(task(), key="solve")
        assert resumed.ok, resumed.errors
        assert resumed.resumed_from == failed.id
        assert FakeRuntime.made[-1].source == "ckpt-2"
        assert EVENTS == ["restore"]
        assert roles(model.requests[0]["messages"]) == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert model.requests[0]["messages"][-1]["content"] == RETRY_NOTE
        assert resumed.labels["call"]["key"] == "agent/solve"
        kept = await agent.run(task(), key="solve")
    assert kept.kept and kept.id == resumed.id


async def test_an_unfinished_trace_in_someone_elses_store_resumes(model):
    """The store is not the caller's: a trace saved mid-run (by prime-rl's live
    writer, here a record taken in `on_save`) is found and continued."""
    saved: list[dict] = []
    first_store = vf.Durable(Partial())
    agent = make(model, save="per_call")
    with vf.use_durable(first_store):
        await agent.run(
            task(),
            key="solve",
            on_save=lambda t, c: saved.append(json.loads(json.dumps(t.to_record()))),
        )
    partial = saved[1]
    assert not partial["is_completed"]
    model.requests.clear()
    with vf.use_durable(vf.Durable(Partial(partial))):
        resumed = await agent.run(task(), key="solve")
    assert resumed.ok, resumed.errors
    assert resumed.resumed_from == partial["id"]
    assert FakeRuntime.made[-1].source == "ckpt-2"


async def test_a_call_whose_box_cannot_be_restored_runs_fresh(
    model, store, monkeypatch
):
    FakeRuntime.doomed["b.txt"] = box_lost
    with vf.use_durable(vf.Durable(store)):
        failed = await make(model, save="per_call").run(task(), key="solve")
        assert failed.checkpoints
        monkeypatch.setattr(
            "verifiers.v1.agent.restore_config", vf.runtimes.restore_config
        )
        fresh = await make(model).run(task(), key="solve")
    assert fresh.ok and fresh.resumed_from is None


# -- interaction(key=) ----------------------------------------------------------


async def drive(agent, prompt=None, messages=("first", "second"), **kwargs):
    replies = []
    async with agent.interaction(task(prompt), key="talk", **kwargs) as exchange:
        for message in messages:
            replies.append((await exchange.turn(message)).last_reply)
    return exchange, replies


async def test_a_finished_exchange_replays_to_the_same_driver(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        first, replies = await drive(agent)
        assert replies == ["done: first", "done: second"]
        assert first.trace.ok and not first.kept
        model.requests.clear()
        again, replayed = await drive(agent)
    assert replayed == replies
    assert again.kept and again.trace.id == first.trace.id
    assert model.requests == []


async def test_an_exchange_is_durable_by_its_position_too(model, store):
    agent = make(model)

    async def driver() -> tuple[vf.Trace, list[str]]:
        replies = []
        async with agent.interaction(task(None)) as exchange:
            for message in ("first", "second"):
                replies.append((await exchange.turn(message)).last_reply)
        return exchange.trace, replies

    with vf.use_durable(vf.Durable(store)):
        first, replies = await driver()
    assert first.labels["call"]["key"] == "agent/0"
    model.requests.clear()
    with vf.use_durable(vf.Durable(store)):
        again, replayed = await driver()
    assert replayed == replies and again.id == first.id and again.kept
    assert model.requests == []


async def test_a_driver_that_diverges_continues_live_from_a_turn_end(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        first, _ = await drive(agent)
        model.requests.clear()
        again, replies = await drive(agent, messages=("first", "changed"))
    assert replies == ["done: first", "done: changed"]
    assert not again.kept
    assert again.trace.resumed_from == first.trace.id
    sent = model.requests[0]["messages"]
    assert [m["content"] for m in sent if m["role"] == "user"] == ["first", "changed"]
    # It continued on the box as the first turn left it.
    assert FakeRuntime.made[-1].files() == {"a.txt": b"a.txt", "b.txt": b"b.txt"}


async def test_a_diverging_driver_without_a_checkpoint_raises(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        await drive(agent)
        with pytest.raises(RuntimeError, match="deterministic"):
            await drive(agent, messages=("first", "changed"))


async def test_a_cut_off_exchange_replays_then_continues_at_its_checkpoint(
    model, store
):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        async with agent.interaction(task(), key="talk") as exchange:
            await exchange.turn()
        failed = exchange.trace
        assert not failed.ok and len(failed.checkpoints) == 2
        model.requests.clear()
        async with agent.interaction(task(), key="talk") as exchange:
            segment = await exchange.turn()
            assert segment.last_reply == f"done: {RETRY_NOTE}"
            assert exchange.live is not None
            segment = await exchange.turn("more")
            assert segment.last_reply == "done: more"
        resumed = exchange.trace
        assert resumed.ok, resumed.errors
        assert resumed.resumed_from == failed.id
        assert roles(model.requests[0]["messages"]) == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        # Finished now: the same driver replays it.
        model.requests.clear()
        async with agent.interaction(task(), key="talk") as exchange:
            assert (await exchange.turn()).last_reply == f"done: {RETRY_NOTE}"
            assert (await exchange.turn("more")).last_reply == "done: more"
        assert exchange.kept and model.requests == []


class FinalizeOnce(NotesTask):
    """Its finalize fails while `failures` lasts, then succeeds."""

    failures: ClassVar[int] = 0
    finalized: ClassVar[int] = 0

    async def finalize(self, trace, runtime) -> None:
        if FinalizeOnce.failures:
            FinalizeOnce.failures -= 1
            raise RuntimeError("finalize broke")
        FinalizeOnce.finalized += 1


async def test_a_failed_finished_exchange_goes_live_instead_of_kept(model, store):
    FinalizeOnce.failures, FinalizeOnce.finalized = 1, 0
    agent = make(model, save="per_call")
    talk = FinalizeOnce(vf.TaskData(idx=0, prompt=None))
    with vf.use_durable(vf.Durable(store)):
        async with agent.interaction(talk, key="talk") as exchange:
            await exchange.turn("hello")
        failed = exchange.trace
        assert failed.is_completed and not failed.ok
        model.requests.clear()
        async with agent.interaction(talk, key="talk") as exchange:
            segment = await exchange.turn("hello")
    # Not given back as an ok, unscored copy: the last turn ran again from its
    # checkpoint, and the live exchange finalized.
    again = exchange.trace
    assert not exchange.kept and not again.kept
    assert again.ok and again.errors == [], again.errors
    assert again.resumed_from == failed.id
    assert len(model.requests) == 1 and segment.last_reply == f"done: {RETRY_NOTE}"
    assert FinalizeOnce.finalized == 1


async def test_a_driver_that_raises_leaves_its_failed_trace(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        with pytest.raises(KeyError):
            async with agent.interaction(task(None), key="talk") as exchange:
                await exchange.turn("hello")
                raise KeyError("driver bug")
        failed = exchange.trace
        stored = store.get(failed.labels["call"]["ref"])
        # The store holds the failed exchange, not its last per-call save (which a
        # rerun would take for one cut off after its last turn, and keep).
        assert stored is not None and stored.id == failed.id
        assert stored.is_completed and not stored.ok
        model.requests.clear()
        async with agent.interaction(task(None), key="talk") as exchange:
            await exchange.turn("hello")
    assert not exchange.kept and exchange.trace.ok
    assert exchange.trace.resumed_from == failed.id and len(model.requests) == 1


async def test_a_replaying_exchange_steers_only_when_live(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        await drive(agent)
        async with agent.interaction(task(None), key="talk") as exchange:
            with pytest.raises(RuntimeError, match="live exchange"):
                await exchange.steer("hey")


# -- continuations (resume=) ----------------------------------------------------


async def test_a_continuation_is_kept_after_a_crash(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        origin = await agent.run(task(), key="draft")
        model.requests.clear()
        first = await agent.run(resume=origin, note="again", key="review")
        assert first.ok, first.errors
        assert not first.kept and first.resumed_from == origin.id
        assert first.labels["call"]["key"] == "agent/review"
        assert "resume" in first.labels["call"]["parts"]
        assert len(model.requests) == 1
        again = await agent.run(resume=origin, note="again", key="review")
    assert again.kept and again.id == first.id
    assert len(model.requests) == 1


async def test_a_continuation_is_durable_by_its_position_too(model, store):
    agent = make(model, save="per_call")

    async def stages() -> list[vf.Trace]:
        with vf.use_durable(vf.Durable(store)):
            origin = await agent.run(task())  # agent/0
            return [origin, await agent.run(resume=origin, note="again")]  # agent/1

    first = await stages()
    assert [t.labels["call"]["key"] for t in first] == ["agent/0", "agent/1"]
    model.requests.clear()
    again = await stages()
    assert all(t.kept for t in again) and model.requests == []
    assert [t.id for t in again] == [t.id for t in first]


async def test_a_cut_off_continuation_resumes_from_its_own_checkpoint(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        origin = await agent.run(task(), key="draft")
        assert [c.id for c in origin.checkpoints] == ["ckpt-1", "ckpt-2", "ckpt-3"]
        FakeRuntime.doomed["b.txt"] = box_lost
        cut = await agent.run(resume=origin, at=0, note="again", key="review")
        assert not cut.ok and cut.resumed_from == origin.id
        # The box it started from (the origin's), then its own per-call saves.
        assert [c.id for c in cut.checkpoints] == ["ckpt-1", "ckpt-4", "ckpt-5"]
        model.requests.clear()
        EVENTS.clear()
        resumed = await agent.run(resume=origin, at=0, note="again", key="review")
    assert resumed.ok, resumed.errors
    # Its own trace and latest checkpoint, not the origin's.
    assert resumed.resumed_from == cut.id
    assert FakeRuntime.made[-1].source == "ckpt-5"
    assert EVENTS == ["restore"]
    sent = model.requests[0]["messages"]
    assert roles(sent) == ["user", "user", "assistant", "tool", "user"]
    assert [m["content"] for m in sent if m["role"] == "user"] == [
        "start",
        "again",
        RETRY_NOTE,
    ]


async def test_a_continuation_of_another_origin_runs_fresh(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        origin = await agent.run(task(), key="draft")
        first = await agent.run(resume=origin, note="again", key="review")
        # The earlier stage ran again (another trace): its continuation reruns.
        rerun = await agent.run(task(), key="redraft")
        fresh = await agent.run(resume=rerun, note="again", key="review")
        assert not fresh.kept and fresh.id != first.id
        assert fresh.resumed_from == rerun.id
        # Another note is another continuation too.
        noted = await agent.run(resume=rerun, note="other", key="review")
        assert not noted.kept and noted.last_reply == "done: other"
        # The latest one of the key is kept.
        assert (await agent.run(resume=rerun, note="other", key="review")).kept


async def test_a_durable_continuation_still_refuses_a_checkpoint(model, store):
    with vf.use_durable(vf.Durable(store)), pytest.raises(TypeError, match="itself"):
        await make(model).run(task(), key="solve", checkpoint="ckpt-1")


async def test_an_exchange_continuing_a_trace_replays_after_a_crash(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        async with agent.interaction(task(None), key="round1") as one:
            await one.turn("first")
        origin = one.trace

        async def round_two():
            # The origin ends on the agent's reply: the first turn continues it.
            async with agent.interaction(resume=origin, key="round2") as two:
                replies = [(await two.turn(m)).last_reply for m in ("second", "third")]
            return two, replies

        model.requests.clear()
        two, replies = await round_two()
        assert replies == ["done: second", "done: third"]
        assert two.trace.resumed_from == origin.id and not two.kept
        sent = model.requests[0]["messages"]
        assert [m["content"] for m in sent if m["role"] == "user"] == [
            "first",
            "second",
        ]
        model.requests.clear()
        again, replayed = await round_two()
    assert again.kept and again.trace.id == two.trace.id
    assert replayed == replies and model.requests == []


async def test_a_cut_off_exchange_continuation_resumes_its_own_trace(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        origin = await agent.run(task(), key="draft")
        FakeRuntime.doomed["b.txt"] = box_lost
        async with agent.interaction(
            resume=origin, at=0, note="again", key="talk"
        ) as exchange:
            await exchange.turn()
        cut = exchange.trace
        assert not cut.ok and cut.resumed_from == origin.id
        model.requests.clear()
        async with agent.interaction(
            resume=origin, at=0, note="again", key="talk"
        ) as exchange:
            segment = await exchange.turn()
            assert segment.last_reply == f"done: {RETRY_NOTE}"
        resumed = exchange.trace
    assert resumed.ok, resumed.errors
    assert resumed.resumed_from == cut.id
    assert FakeRuntime.made[-1].source == "ckpt-5"
    assert roles(model.requests[0]["messages"]) == [
        "user",
        "user",
        "assistant",
        "tool",
        "user",
    ]


# -- the box during a replay -------------------------------------------------------


@pytest.fixture
def restorable_boxes(monkeypatch) -> list[str | None]:
    """The boxes a replay restores (`provision_box`) are fake checkpointing boxes too;
    returns the checkpoints they were made from, in order."""
    sources: list[str | None] = []

    def fake(config, name=None):
        sources.append(getattr(config, "checkpoint", None))
        return FakeRuntime(config, name)

    def restore(config, checkpoint):
        return RestoreConfig(**config.model_dump(), checkpoint=checkpoint)

    monkeypatch.setattr(runtimes_module, "make_runtime", fake)
    monkeypatch.setattr(box_module, "make_runtime", fake)
    monkeypatch.setattr(box_module, "restore_config", restore)
    return sources


async def read_or_none(runtime, path: str) -> bytes | None:
    try:
        return await runtime.read(path)
    except vf.SandboxError:
        return None


async def logging_driver(agent, messages, seen, held=None):
    """Reads the log the previous turn left in the box after each turn, then logs
    this turn there: host code that reads and writes the box between turns."""
    async with agent.interaction(task(None), key="talk") as exchange:
        for message in messages:
            await exchange.turn(message)
            seen.append(await read_or_none(exchange.runtime, "log.txt"))
            await exchange.runtime.write("log.txt", f"after {message}".encode())
            if held is not None:
                held.append(await exchange.box())
    return exchange


async def test_a_replay_reads_the_box_each_turn_left(model, store, restorable_boxes):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        live: list = []
        first = await logging_driver(agent, ("first", "second"), live)
        assert first.trace.ok and live == [None, b"after first"]
        model.requests.clear()
        restorable_boxes.clear()
        replayed: list = []
        held: list = []
        again = await logging_driver(agent, ("first", "second"), replayed, held)
    assert again.kept and model.requests == []
    # Each read saw the box as that turn left it, restored from its checkpoint.
    assert replayed == live
    assert restorable_boxes == ["ckpt-3", "ckpt-4"]
    # Moved in place: a reference to the box stays the box.
    assert held[0] is held[1]
    assert held[0].stopped


async def test_the_live_exchange_continues_on_the_box_the_replay_restored(
    model, store, restorable_boxes
):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        await logging_driver(agent, ("first", "second"), [])
        model.requests.clear()
        restorable_boxes.clear()
        seen: list = []
        exchange = await logging_driver(agent, ("first", "second", "third"), seen)
    assert seen == [None, b"after first", b"after second"]
    # Live from the third turn, on the box restored for the second (no new box): what
    # the driver wrote to it after the second turn was still there.
    assert not exchange.kept and exchange.live is not None
    assert exchange.live.trace.agent.runtime.borrowed
    assert restorable_boxes == ["ckpt-3", "ckpt-4"]
    assert [m["messages"][-1]["content"] for m in model.requests] == ["third"]


async def test_a_replay_that_never_reads_restores_nothing(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        await drive(agent)
        FakeRuntime.made.clear()
        async with agent.interaction(task(None), key="talk") as exchange:
            await exchange.turn("first")
            await exchange.turn("second")
            assert isinstance(exchange.runtime, vf.ReplayBox)
            assert FakeRuntime.made == []
            # Past the recorded turns, the live exchange restores its own box.
            assert (await exchange.turn("third")).last_reply == "done: third"
            assert [box.source for box in FakeRuntime.made] == ["ckpt-4"]


async def test_a_replay_without_turn_end_checkpoints_cannot_read(
    model, store, restorable_boxes
):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        await drive(agent)
        async with agent.interaction(task(None), key="talk") as exchange:
            await exchange.turn("first")
            with pytest.raises(RuntimeError, match="checkpoint"):
                await exchange.runtime.read("a.txt")


# -- a continuation cut off before it moved ----------------------------------------


def tool_each_turn(messages: list[dict]) -> dict:
    """The scripted model, one tool call per turn: each user turn writes
    `<turn>.txt`, then answers it."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    if messages[-1]["role"] == "user":
        name = f"{len(users)}.txt"
        arguments = json.dumps({"path": name, "text": name})
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{len(messages)}",
                    "type": "function",
                    "function": {"name": "write", "arguments": arguments},
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": f"done: {users[-1]}"}
        finish = "stop"
    return {
        "id": "scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "scripted",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class Records:
    """The records a run saved (prime-rl's live writer), by trace: `saves` at each
    per-call save, `minted` as each trace was minted."""

    def __init__(self) -> None:
        self.saves: list[dict] = []
        self.minted: list[dict] = []

    def on_save(self, trace, checkpoint) -> None:
        self.saves.append(json.loads(json.dumps(trace.to_record())))

    def on_trace(self, trace) -> None:
        self.minted.append(json.loads(json.dumps(trace.to_record())))


async def box_reading_driver(agent, messages, seen, records=None, **kwargs):
    """Reads the log in the box before the first turn and after each turn, then logs
    the turn there (hillclimb's driver sets its task up on the box first)."""
    hooks = (
        {"on_save": records.on_save, "on_trace": records.on_trace}
        if records is not None
        else {}
    )
    async with agent.interaction(key="talk", **hooks, **kwargs) as exchange:
        seen.append(await read_or_none(exchange.runtime, "log.txt"))
        for message in messages:
            await exchange.turn(message)
            seen.append(await read_or_none(exchange.runtime, "log.txt"))
            await exchange.runtime.write("log.txt", f"after {message}".encode())
    return exchange


async def cut_twice(agent, messages, progress: bool) -> tuple[list, dict, dict]:
    """A durable exchange cut off inside its third turn (at that turn's tool-call
    save), continued, and cut off again: before its first model call finished, or,
    with `progress`, after the continued turn's final reply was saved. Returns what
    the uncut exchange read from its box, and the records of the two cuts."""
    seen: list = []
    first = Records()
    with vf.use_durable(vf.Durable(Partial())):
        await box_reading_driver(agent, messages, seen, first, task=task(None))
    assert [r["checkpoints"][-1]["id"] for r in first.saves] == [
        f"ckpt-{n}" for n in range(1, 7)
    ]
    cut = first.saves[4]  # the third turn's tool-call reply
    assert not cut["is_completed"]
    second = Records()
    with vf.use_durable(vf.Durable(Partial(cut))):
        await box_reading_driver(agent, messages, [], second, task=task(None))
    [minted] = second.minted
    again = second.saves[-1] if progress else minted
    return seen, cut, again


async def test_a_continuation_cut_before_it_moved_restores_the_box_anywhere(
    model, restorable_boxes, monkeypatch
):
    """The production chain: a continuation minted from a cut-off exchange (its
    conversation in its prompt, one checkpoint at its end) was cut off before its first
    model call. Replaying it restores the box before the first turn and between the
    replayed turns from the checkpoints it carried from the exchange it continued."""
    monkeypatch.setattr(resume_tests, "reply", tool_each_turn)
    agent = make(model, save="per_call")
    messages = ("first", "second", "third")
    seen, cut, minted = await cut_twice(agent, messages, progress=False)
    assert minted["nodes"] == [] and minted["resumed_from"] == cut["id"]
    # The cut exchange's checkpoints inside the minted conversation, at their places
    # in it, then the one it continues from.
    assert [
        (c["id"], c.get("position"), c.get("restored", False))
        for c in minted["checkpoints"]
    ] == [
        ("ckpt-1", 1, False),
        ("ckpt-2", 5, False),
        ("ckpt-3", 5, False),
        ("ckpt-4", None, True),
    ]
    restorable_boxes.clear()
    model.requests.clear()
    replayed: list = []
    with vf.use_durable(vf.Durable(Partial(cut, minted))):
        exchange = await box_reading_driver(agent, messages, replayed, task=task(None))
    # Each read saw the box as the uncut exchange had it there.
    assert replayed == seen == [None, None, b"after first", b"after second"]
    # Before the first turn, then after each replayed turn; the live turn continues
    # on the box as the second turn left it, where the minted trace starts.
    assert restorable_boxes == ["ckpt-1", "ckpt-2", "ckpt-4"]
    assert exchange.live is not None and exchange.trace.ok
    assert exchange.trace.resumed_from == minted["id"]


async def test_a_continuation_that_moved_restores_the_box_anywhere(
    model, restorable_boxes, monkeypatch
):
    monkeypatch.setattr(resume_tests, "reply", tool_each_turn)
    agent = make(model, save="per_call")
    seen, cut, saved = await cut_twice(
        agent, ("first", "second", "third"), progress=True
    )
    assert saved["nodes"] and saved["resumed_from"] == cut["id"]
    assert [(c["id"], c.get("position")) for c in saved["checkpoints"]] == [
        ("ckpt-1", 1),
        ("ckpt-2", 5),
        ("ckpt-3", 5),
        ("ckpt-4", None),
        ("ckpt-7", None),
        ("ckpt-8", None),
    ]
    restorable_boxes.clear()
    model.requests.clear()
    replayed: list = []
    with vf.use_durable(vf.Durable(Partial(cut, saved))):
        exchange = await box_reading_driver(
            agent, ("first", "second", "third", "fourth"), replayed, task=task(None)
        )
    assert replayed == [*seen, b"after third"]
    assert restorable_boxes == ["ckpt-1", "ckpt-2", "ckpt-4", "ckpt-8"]
    # Live from the fourth turn only.
    assert [m["messages"][-1]["content"] for m in model.requests] == ["fourth", "ok"]
    assert exchange.trace.ok and exchange.trace.resumed_from == saved["id"]
    # The next link of the chain carries what this record carried, and its own.
    assert [(c.id, c.position) for c in exchange.trace.checkpoints] == [
        ("ckpt-1", 1),
        ("ckpt-2", 5),
        ("ckpt-3", 5),
        ("ckpt-4", 9),
        ("ckpt-7", 9),
        ("ckpt-8", None),
        ("ckpt-9", None),
        ("ckpt-10", None),
    ]


async def test_a_record_carrying_nothing_replays_from_the_calls_opening(
    model, restorable_boxes, monkeypatch
):
    """A continuation recorded before checkpoints were carried (no `position`) loads
    and replays as before; before its first turn the box is the one the call opened
    on (its origin's checkpoint `at`)."""
    monkeypatch.setattr(resume_tests, "reply", tool_each_turn)
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(Partial())):
        async with agent.interaction(task(None), key="draft") as draft:
            await draft.turn("draft")
            await draft.runtime.write("log.txt", b"drafted")
            opened_on = await draft.checkpoint()
    origin = draft.trace
    messages = ("first", "second")
    first = Records()
    with vf.use_durable(vf.Durable(Partial())):
        await box_reading_driver(agent, messages, [], first, resume=origin)
    # The second turn's tool-call reply (the call opened on `opened_on`).
    cut = first.saves[2]
    assert [c["id"] for c in cut["checkpoints"] if c.get("restored")] == [opened_on.id]
    second = Records()
    with vf.use_durable(vf.Durable(Partial(cut))):
        await box_reading_driver(agent, messages, [], second, resume=origin)
    [minted] = second.minted
    # As written before checkpoints were carried: the one it continues from only.
    old = {
        **minted,
        "checkpoints": [c for c in minted["checkpoints"] if "position" not in c],
    }
    assert len(old["checkpoints"]) == 1 and old["checkpoints"][0]["restored"]
    assert vf.WireTrace.model_validate(old).checkpoints[0].position is None
    restorable_boxes.clear()
    model.requests.clear()
    replayed: list = []
    with vf.use_durable(vf.Durable(Partial(cut, old))):
        async with agent.interaction(resume=origin, key="talk") as exchange:
            replayed.append(await read_or_none(exchange.runtime, "log.txt"))
            await exchange.turn("first")
            await exchange.turn("second")
    assert replayed == [b"drafted"]
    assert restorable_boxes[0] == opened_on.id
    # The first turn replayed; the second continued live from the cut, as before.
    assert [m["messages"][-1]["content"] for m in model.requests] == ["second", "ok"]
    assert exchange.trace.ok and exchange.trace.resumed_from == old["id"]


# -- an episode -----------------------------------------------------------------


class PairConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig(
        runtime={"type": "subprocess"}, harness={"id": "null"}, save="per_call"
    )


class PairEnv(vf.Env[PairConfig]):
    """Two calls, durable by position; crashes between them while `crash` is set."""

    root: Path
    crash = True

    def trace_labels(self, task):
        return {"job": "j1"}

    def durable(self, task):
        return vf.Durable(vf.TraceDir(self.root), match=self.trace_labels(task))

    async def run(self, task, agents):
        await agents.solver.run(task)
        if PairEnv.crash:
            raise RuntimeError("the env server died")
        await agents.solver.run(NotesTask(vf.TaskData(idx=1, prompt="two")))


async def test_an_episode_run_again_skips_its_finished_calls(
    model, tmp_path, monkeypatch
):
    from verifiers.v1.utils import loaders

    load = loaders.load_harness
    monkeypatch.setattr(
        loaders,
        "load_harness",
        lambda config: ScriptHarness(config) if config.id == "null" else load(config),
    )
    PairEnv.root, PairEnv.crash = tmp_path / "traces", True
    env = PairEnv(PairConfig(taskset=vf.TasksetConfig(id="echo-v1")))
    ctx = ModelContext(
        model="scripted",
        client=EvalClientConfig(base_url=model.url, api_key_var="SCRIPTED_KEY"),
    )
    crashed = await env.run_episode(task(), ctx)
    assert not crashed.ok and len(crashed.traces) == 1
    assert crashed.traces[0].labels["job"] == "j1"
    assert crashed.traces[0].labels["call"]["key"] == "solver/0"
    PairEnv.crash = False
    model.requests.clear()
    episode = await env.run_episode(task(), ctx)
    assert episode.ok, episode.errors
    # The finished call was kept: the episode holds it and ran only the second.
    first, second = episode.traces
    assert first.kept and first.id == crashed.traces[0].id
    assert second.labels["call"]["key"] == "solver/1" and not second.kept
    assert all(r["messages"][0]["content"] == "two" for r in model.requests)
    # Everything kept: the episode ran nothing and is still whole.
    model.requests.clear()
    again = await env.run_episode(task(), ctx)
    assert again.ok, again.errors
    assert [t.id for t in again.traces] == [first.id, second.id]
    assert all(t.kept for t in again.traces) and model.requests == []


class TalkEnv(vf.Env[PairConfig]):
    """One exchange of two turns; crashes after its last turn, before it closes,
    while `crash` is set (its server dies: the episode is cancelled). The driver notes
    a verdict on the trace once it closed."""

    root: Path
    crash = True

    def trace_labels(self, task):
        return {"job": "j1"}

    def durable(self, task):
        return vf.Durable(vf.TraceDir(self.root), match=self.trace_labels(task))

    async def run(self, task, agents):
        async with agents.solver.interaction(task) as exchange:
            for message in ("first", "second"):
                await exchange.turn(message)
            if TalkEnv.crash:
                raise asyncio.CancelledError()
        exchange.trace.info["verdict"] = "pass"


async def test_an_exchange_cut_after_its_last_turn_replays_into_the_episode(
    model, tmp_path, monkeypatch
):
    from verifiers.v1.utils import loaders

    load = loaders.load_harness
    monkeypatch.setattr(
        loaders,
        "load_harness",
        lambda config: ScriptHarness(config) if config.id == "null" else load(config),
    )
    TalkEnv.root, TalkEnv.crash = tmp_path / "traces", True
    env = TalkEnv(PairConfig(taskset=vf.TasksetConfig(id="echo-v1")))
    ctx = ModelContext(
        model="scripted",
        client=EvalClientConfig(base_url=model.url, api_key_var="SCRIPTED_KEY"),
    )
    with pytest.raises(asyncio.CancelledError):
        await env.run_episode(task(None), ctx)
    cut = vf.TraceDir(TalkEnv.root).get(call_ref({"job": "j1"}, "solver/0"))
    assert cut is not None and not cut.is_completed
    TalkEnv.crash = False
    model.requests.clear()
    # The driver gives back exactly the recorded turns, then closes: nothing goes live.
    episode = await env.run_episode(task(None), ctx)
    assert episode.ok, episode.errors
    assert model.requests == []
    [kept] = episode.traces
    assert kept.kept and kept.id == cut.id
    assert kept.is_completed and kept.ok and kept.stop_condition == "user_closed"
    [landed] = episode.to_record()["traces"]
    assert landed["id"] == cut.id and landed["info"]["verdict"] == "pass"
    # The store holds it closed: the next episode replays it as a finished exchange.
    stored = vf.TraceDir(TalkEnv.root).get(kept.labels["call"]["ref"])
    assert stored is not None and stored.is_completed and stored.ok
    again = await env.run_episode(task(None), ctx)
    assert again.ok and model.requests == []
    assert [t.id for t in again.traces] == [cut.id] and again.traces[0].kept


class ScopedEnv(vf.Env[PairConfig]):
    """One solver run per episode, scoped by `labels`; it waits for `release` while
    `hold` is set, so a test can start another episode beside it."""

    root: Path
    labels: ClassVar[dict] = {"job": "j1"}
    hold: ClassVar[bool] = False
    entered: ClassVar[asyncio.Event]
    release: ClassVar[asyncio.Event]

    def trace_labels(self, task):
        return dict(ScopedEnv.labels)

    def durable(self, task):
        return vf.Durable(vf.TraceDir(self.root))

    async def run(self, task, agents):
        if ScopedEnv.hold:
            ScopedEnv.entered.set()
            await ScopedEnv.release.wait()
        await agents.solver.run(task)


@pytest.fixture
def scoped(model, tmp_path, monkeypatch):
    from verifiers.v1.utils import loaders

    load = loaders.load_harness
    monkeypatch.setattr(
        loaders,
        "load_harness",
        lambda config: ScriptHarness(config) if config.id == "null" else load(config),
    )
    ScopedEnv.root, ScopedEnv.labels, ScopedEnv.hold = tmp_path, {"job": "j1"}, False
    ScopedEnv.entered, ScopedEnv.release = asyncio.Event(), asyncio.Event()
    env = ScopedEnv(PairConfig(taskset=vf.TasksetConfig(id="echo-v1")))
    ctx = ModelContext(
        model="scripted",
        client=EvalClientConfig(base_url=model.url, api_key_var="SCRIPTED_KEY"),
    )
    return env, ctx


async def test_a_durable_episode_needs_a_scope(model, scoped):
    env, ctx = scoped
    ScopedEnv.labels = {}
    episode = await env.run_episode(task(), ctx)
    assert not episode.ok and episode.traces == [] and model.requests == []
    assert "scope that identifies it" in episode.errors[-1].message


async def test_two_live_episodes_cannot_share_a_scope(model, scoped):
    env, ctx = scoped
    ScopedEnv.hold = True
    first = asyncio.create_task(env.run_episode(task(), ctx))
    await ScopedEnv.entered.wait()
    ScopedEnv.hold = False  # only the first waits
    # A second rollout of the task, labelled the same, would take the first's calls.
    second = await env.run_episode(task(), ctx)
    assert not second.ok and second.traces == []
    assert "is running" in second.errors[-1].message
    ScopedEnv.release.set()
    assert (await first).ok
    # Once the first landed, the scope is free: run again, it keeps its call.
    again = await env.run_episode(task(), ctx)
    assert again.ok and again.traces[0].kept


# -- a multi-agent episode, crashed and run again ---------------------------------


class StudioConfig(vf.EnvConfig):
    writer: vf.AgentConfig = vf.AgentConfig(
        runtime={"type": "subprocess"}, harness={"id": "null"}, save="per_call"
    )
    judge: vf.AgentConfig = vf.AgentConfig(
        runtime={"type": "subprocess"}, harness={"id": "null"}, save="per_call"
    )


class StudioEnv(vf.Env[StudioConfig]):
    """A writer drafts, host code derives the judge's brief from the draft and a seeded
    draw, the judge reviews it. The scope is the env's trace labels, the lookup a
    directory of traces; every call is durable by its seat and position."""

    root: Path
    precheck = False
    """A later version of the code: a judge call inserted before the review."""
    host: ClassVar[list[str]] = []
    """What the host code between the calls computed, attempt by attempt."""

    def trace_labels(self, task):
        return {"job": "j1", "stage": "review"}

    def durable(self, task):
        return vf.Durable(vf.TraceDir(self.root))

    async def run(self, task, agents):
        draft = await agents.writer.run(task)
        if not draft.ok:
            raise RuntimeError("the draft failed")
        pick = vf.episode_random().randrange(10**6)
        brief = f"review {draft.last_reply!r} #{pick}"
        StudioEnv.host.append(brief)
        if StudioEnv.precheck:
            await agents.judge.run(NotesTask(vf.TaskData(idx=1, prompt="precheck")))
        review = await agents.judge.run(NotesTask(vf.TaskData(idx=1, prompt=brief)))
        if not review.ok:
            raise RuntimeError("the review failed")


@pytest.fixture
def studio(model, tmp_path, monkeypatch):
    from verifiers.v1.utils import loaders

    load = loaders.load_harness
    monkeypatch.setattr(
        loaders,
        "load_harness",
        lambda config: ScriptHarness(config) if config.id == "null" else load(config),
    )
    StudioEnv.root, StudioEnv.precheck, StudioEnv.host = tmp_path / "traces", False, []
    env = StudioEnv(StudioConfig(taskset=vf.TasksetConfig(id="echo-v1")))
    ctx = ModelContext(
        model="scripted",
        client=EvalClientConfig(base_url=model.url, api_key_var="SCRIPTED_KEY"),
    )
    return env, ctx


async def crash_in_the_judges_second_write(env, ctx, monkeypatch):
    """Run the episode until the judge's second tool call, then kill it there, as a
    dying env server would: the judge's trace is left as its last per-call save."""
    original = FakeRuntime.write
    episode = asyncio.current_task()
    writes = {"b.txt": 0}

    async def write(self, path, data):
        if path == "b.txt":
            writes["b.txt"] += 1
            if writes["b.txt"] == 2:  # the writer's is the first
                episode.cancel()
                await asyncio.Event().wait()
        await original(self, path, data)

    monkeypatch.setattr(FakeRuntime, "write", write)
    try:
        with pytest.raises(asyncio.CancelledError):
            await env.run_episode(task(), ctx)
    finally:
        monkeypatch.setattr(FakeRuntime, "write", original)
        episode.uncancel()


async def test_a_crashed_episode_reruns_from_the_top_on_its_traces(
    model, studio, monkeypatch
):
    env, ctx = studio
    await asyncio.create_task(crash_in_the_judges_second_write(env, ctx, monkeypatch))
    store = vf.TraceDir(StudioEnv.root)
    scope = env.trace_labels(task())
    writer = store.get(call_ref(scope, "writer/0"))
    cut = store.get(call_ref(scope, "judge/0"))
    assert writer is not None and writer.ok and writer.labels["stage"] == "review"
    assert cut is not None and not cut.is_completed and len(cut.checkpoints) == 2
    model.requests.clear()
    EVENTS.clear()

    episode = await env.run_episode(task(), ctx)
    assert episode.ok, episode.errors
    # The host code between the calls computed the same brief from the kept draft
    # and the same seeded draw.
    assert len(StudioEnv.host) == 2 and StudioEnv.host[0] == StudioEnv.host[1]
    # The writer ran nothing; the judge continued from its latest checkpoint.
    draft, review = episode.traces
    assert draft.kept and draft.id == writer.id
    assert review.labels["call"]["key"] == "judge/0" and not review.kept
    assert review.resumed_from == cut.id
    assert EVENTS == ["restore"]
    assert len(model.requests) == 2
    sent = model.requests[0]["messages"]
    assert sent[0]["content"] == StudioEnv.host[0]
    assert sent[-1]["content"] == RETRY_NOTE

    # Everything finished: run again, nothing runs.
    model.requests.clear()
    again = await env.run_episode(task(), ctx)
    assert again.ok and model.requests == []
    assert [t.id for t in again.traces] == [draft.id, review.id]
    assert StudioEnv.host[2] == StudioEnv.host[0]


async def test_a_code_change_that_shifts_positions_reruns_them(model, studio):
    env, ctx = studio
    first = await env.run_episode(task(), ctx)
    assert first.ok, first.errors
    [old_review] = [t for t in first.traces if t.labels["call"]["key"] == "judge/0"]
    model.requests.clear()

    # The new code inserts a judge call: the review moves to judge/1, and judge/0
    # now holds another prompt, so its fingerprint no longer matches the old trace.
    StudioEnv.precheck = True
    episode = await env.run_episode(task(), ctx)
    assert episode.ok, episode.errors
    by_key = {t.labels["call"]["key"]: t for t in episode.traces}
    assert sorted(by_key) == ["judge/0", "judge/1", "writer/0"]
    assert by_key["writer/0"].kept
    precheck, review = by_key["judge/0"], by_key["judge/1"]
    assert precheck.last_reply == "done: precheck" and precheck.resumed_from is None
    assert review.id != old_review.id and review.resumed_from is None
    assert review.last_reply == old_review.last_reply
    # The writer's call kept its position and fingerprint.
    assert all(
        m["messages"][0]["content"] in ("precheck", StudioEnv.host[-1])
        for m in model.requests
    )
    model.requests.clear()
    again = await env.run_episode(task(), ctx)
    assert again.ok and model.requests == []
    assert all(t.kept for t in again.traces) and len(again.traces) == 3


async def test_a_finished_run_streams_its_final_ok(model):
    """The final `ok` is set after the rollout's last phase notify; a watcher (a
    durable-call lookup reading the streamed trace) must still see it."""
    seen: list[tuple[bool, bool]] = []

    def watch(trace):
        trace.watch(lambda t: seen.append((t.is_completed, t.ok)))

    trace = await make(model).run(task(), on_trace=watch)
    assert trace.ok, trace.errors
    assert seen[-1] == (True, True)


def test_a_ref_points_at_the_trace_that_moved_last(tmp_path):
    """Seats on one box save in any order: a write from a run that moved earlier does
    not take the box's ref from the one that moved later."""
    store = vf.TraceDir(tmp_path)

    def run_on_box(start: float, *saves: float) -> vf.Trace:
        trace = vf.Trace(
            task=vf.TraceTask(type="Task", data=vf.TaskData(prompt="hi")),
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            info={"labels": {"box": {"key": "b", "ref": "r"}}},
        )
        trace.timing.boot.start = start
        trace.checkpoints = [vf.Checkpoint(id=f"c{t}", turns=0, time=t) for t in saves]
        return trace

    later, earlier = run_on_box(1.0, 10.0), run_on_box(2.0, 5.0)
    store.write(later)
    store.write(earlier)  # its final write comes last, but it moved first
    assert store.get("r").id == later.id
    earlier.checkpoints.append(vf.Checkpoint(id="c20", turns=1, time=20.0))
    store.write(earlier)
    assert store.get("r").id == earlier.id
    assert store.get("other") is None
