"""Durable calls (`verifiers.v1.durable`): with a durable installed, `agent.run` and
`agent.interaction` find their earlier traces (by position, or by `key=`) through a
`TraceLookup` and return the finished one, continue a cut-off one from its latest
checkpoint, or run fresh; `Env.durable(task)` installs one for an episode.

Deterministic, no API key: the scripted model, in-process harness and fake
checkpointing boxes of `test_agent_resume`.
"""

# ruff: noqa: F811 - the fixtures imported from test_agent_resume are parameters

import asyncio
import json
from pathlib import Path

import pytest
from test_agent_resume import (  # noqa: F401 - fixtures
    EVENTS,
    FakeRuntime,
    NotesTask,
    ScriptHarness,
    box_lost,
    fake_boxes,
    make,
    model,
    roles,
    task,
)

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.durable import call_parts, labels_match, resolve_call
from verifiers.v1.utils.retries import RETRY_NOTE


@pytest.fixture
def store(tmp_path) -> vf.TraceDir:
    return vf.TraceDir(tmp_path / "traces")


class Partial:
    """A lookup over a store someone else writes: here, one trace as saved mid-run."""

    def __init__(self, *records: dict) -> None:
        self.records = list(records)

    def find(self, labels):
        traces = [vf.WireTrace.model_validate(r) for r in self.records]
        return [t for t in traces if labels_match(t.labels, labels)]


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
    assert label["key"] == "solve"
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
    # depends= and cache_inputs= are inert on a plain call.
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
    assert named.labels["call"]["key"] == "named"
    # A named call takes no position.
    assert positioned.labels["call"]["key"] == "agent/0"
    assert plain.info == {}
    with vf.use_durable(vf.Durable(store)):
        assert (await agent.run(task(), key="named")).kept
        assert not (await agent.run(task(), durable=False)).kept
    assert len(model.requests) == 12


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
        assert resolve_call(durable, "solve", parts).kept is None


async def test_depends_and_cache_inputs(model, store, tmp_path):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("v1")
    agent = make(model)
    with vf.use_durable(vf.Durable(store, parts={"version": "1"})):
        await agent.run(task(), key="solve", depends=prompt_file)
        assert (await agent.run(task(), key="solve", depends=prompt_file)).kept
        prompt_file.write_text("v2")
        assert not (await agent.run(task(), key="solve", depends=prompt_file)).kept
        # cache_inputs replaces the task and agent: another prompt keeps it.
        await agent.run(task(), key="pinned", cache_inputs={"n": 1})
        assert (
            await agent.run(task("other"), key="pinned", cache_inputs={"n": 1})
        ).kept
    # The durable's parts are in every fingerprint.
    with vf.use_durable(vf.Durable(store, parts={"version": "2"})):
        assert not (await agent.run(task(), key="pinned", cache_inputs={"n": 1})).kept


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
        assert resumed.labels["call"]["key"] == "solve"
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


async def test_a_replaying_exchange_steers_only_when_live(model, store):
    agent = make(model)
    with vf.use_durable(vf.Durable(store)):
        await drive(agent)
        async with agent.interaction(task(None), key="talk") as exchange:
            with pytest.raises(RuntimeError, match="live exchange"):
                await exchange.steer("hey")


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
    # The finished call joined the first episode; this one ran only the second.
    assert [t.labels["call"]["key"] for t in episode.traces] == ["solver/1"]
    assert all(r["messages"][0]["content"] == "two" for r in model.requests)
    # Everything kept: the episode ran nothing and is still whole.
    model.requests.clear()
    again = await env.run_episode(task(), ctx)
    assert again.ok, again.errors
    assert again.traces == [] and model.requests == []
