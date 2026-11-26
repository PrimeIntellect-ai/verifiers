"""An episode's agent calls are keyed by seat and call number, and a continued episode
replays them from its restore point: a finished call returns its saved trace without
running, a running call continues from its saved trace, and a call that made no turn
runs fresh."""

import asyncio
import pickle
from collections import defaultdict
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Agent, _EpisodeAgent
from verifiers.v1.env import Env, RunSlot
from verifiers.v1.restore import EpisodeCalls, ReplayDiverged, episode_restore_point


def task(prompt: str) -> vf.Task:
    return vf.Task(vf.TaskData(idx=0, prompt=prompt))


def new_trace(prompt: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task", data=vf.TaskData(idx=0, prompt=prompt), key="k", hash="h"
        ),
    )


def saved_trace(trace: vf.Trace) -> dict:
    """`trace` as `save_all` returns it: its dump, with the checkpoint it continues from."""
    data = trace.model_dump(context={"token_arrays": True})
    data["checkpoints"] = [
        {"id": f"c-{trace.id}", "snapshot": {"id": f"s-{trace.id}"}, "session": {}}
    ]
    return data


class Runs:
    """Stands in for the base agent's runs: each mints a trace, runs its scoring hook, and
    finishes when its seat's gate opens."""

    def __init__(self, monkeypatch) -> None:
        self.started: list[tuple[str, str, bool]] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.running: defaultdict[str, asyncio.Event] = defaultdict(asyncio.Event)
        runs = self

        async def run(
            self, handle, task, *, on_trace=None, on_score=None, restore=None, **_
        ):
            trace = (
                vf.Trace.model_validate(restore)
                if restore is not None
                else new_trace(task.data.prompt_text)
            )
            runs.started.append(
                (self._name, task.data.prompt_text, restore is not None)
            )
            on_trace(trace)
            runs.running[self._name].set()
            if (gate := runs.gates.get(self._name)) is not None:
                await gate.wait()
            if on_score is not None:
                await on_score(trace, None)
            trace.ok = True
            return trace

        monkeypatch.setattr(Agent, "_run", run)


def agents(
    calls: EpisodeCalls, completed: list[vf.Trace], *seats: str
) -> dict[str, _EpisodeAgent]:
    return {
        seat: _EpisodeAgent(
            vf.AgentConfig(model="m"),
            interception=None,
            name=seat,
            shared_tools={},
            task_cls=vf.Task,
            gate=None,
            completed=completed,
            on_trace=None,
            on_discard=None,
            warned_resources=set(),
            calls=calls,
        )
        for seat in seats
    }


async def judged(seats: dict[str, _EpisodeAgent]) -> vf.Trace:
    """A solver whose scoring runs a judge, as the agentic judge env does."""

    async def judge(solution, runtime):
        await seats["judge"].run(task(f"judge {solution.task.data.prompt_text}"))

    return await seats["solver"].run(task("solve"), on_score=judge)


@pytest.mark.asyncio
async def test_calls_are_keyed_by_seat_in_the_order_they_start(monkeypatch):
    Runs(monkeypatch)
    calls, completed = EpisodeCalls(), []
    seats = agents(calls, completed, "a", "b", "solver", "judge")
    await asyncio.gather(
        seats["a"].run(task("x")), seats["a"].run(task("y")), seats["b"].run(task("z"))
    )
    await judged(seats)
    keys = {key: call.parent for key, call in calls.calls.items()}
    assert keys == {
        ("a", 0): None,
        ("a", 1): None,
        ("b", 0): None,
        ("solver", 0): None,
        ("judge", 0): ("solver", 0),
    }
    assert [calls.calls[("a", n)].trace.task.data.prompt_text for n in (0, 1)] == [
        "x",
        "y",
    ]


@pytest.mark.asyncio
async def test_a_continued_episode_replays_finished_calls_and_continues_running_ones(
    monkeypatch,
):
    runs = Runs(monkeypatch)
    calls, completed = EpisodeCalls(), []
    seats = agents(calls, completed, "a", "b")
    first = await seats["a"].run(task("first"))
    runs.gates["b"] = asyncio.Event()
    second = asyncio.ensure_future(seats["b"].run(task("second")))
    await runs.running["b"].wait()
    running = calls.calls[("b", 0)].trace
    point, reason = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(running)]
    )
    runs.gates["b"].set()
    await second
    assert reason is None
    point = pickle.loads(pickle.dumps(point))
    assert [(c["seat"], c["n"], c["state"]) for c in point["calls"]] == [
        ("a", 0, "finished"),
        ("b", 0, "running"),
    ]
    assert point["snapshots"] == [{"id": f"s-{running.id}"}]

    runs.started.clear()
    calls, completed = EpisodeCalls.continuing(point), []
    seats = agents(calls, completed, "a", "b")
    replayed = await seats["a"].run(task("first"))
    continued = await seats["b"].run(task("second"))
    # The finished call did not run again; the running one continued its saved trace.
    assert runs.started == [("b", "second", True)]
    assert replayed.id == first.id and continued.id == running.id
    assert [t.id for t in completed] == [first.id, running.id]


@pytest.mark.asyncio
async def test_a_replayed_call_brings_the_calls_its_scoring_made(monkeypatch):
    runs = Runs(monkeypatch)
    calls, completed = EpisodeCalls(), []
    await judged(agents(calls, completed, "solver", "judge"))
    point, _ = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert [(c["seat"], c["parent"]) for c in point["calls"]] == [
        ("solver", None),
        ("judge", ["solver", 0]),
    ]

    runs.started.clear()
    calls, completed = EpisodeCalls.continuing(point), []
    seats = agents(calls, completed, "solver", "judge")
    await judged(seats)
    assert runs.started == []
    # As in the saved episode, the judge's trace joins first.
    assert [t.task.data.prompt_text for t in completed] == ["judge solve", "solve"]
    # The judge's key is taken: a later judge call is its second.
    await seats["judge"].run(task("again"))
    assert ("judge", 1) in calls.calls


@pytest.mark.asyncio
async def test_a_call_asked_for_another_task_stops_the_replay(monkeypatch):
    Runs(monkeypatch)
    calls, completed = EpisodeCalls(), []
    await agents(calls, completed, "a")["a"].run(task("first"))
    point, _ = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    calls = EpisodeCalls.continuing(point)
    with pytest.raises(ReplayDiverged):
        await agents(calls, [], "a")["a"].run(task("changed"))
    assert calls.diverged is not None


def test_a_running_call_without_a_saved_trace_runs_fresh_only_before_its_first_turn():
    calls = EpisodeCalls()
    calls.start("a", task("x"), None)
    calls.minted(("a", 0), SimpleNamespace(id="t-a", num_turns=0))
    point, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert reason is None and point["calls"] == []

    calls.calls[("a", 0)].trace = SimpleNamespace(id="t-a", num_turns=2)
    point, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert point is None and reason == "a call 0: it took no memory snapshot"


def test_an_episode_with_an_interaction_or_a_run_inside_scoring_has_no_restore_point():
    calls = EpisodeCalls()
    calls.start("user", task("x"), None, interaction=True)
    point, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert point is None and "it is an interaction" in reason

    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0))
    calls.minted(("solver", 0), SimpleNamespace(id="t-s", num_turns=3))
    calls.minted(("judge", 0), SimpleNamespace(id="t-j", num_turns=1))
    point, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert point is None and "inside another run's scoring" in reason


def test_a_single_saved_rollout_continues_the_first_call_of_its_seat():
    trace = new_trace("x")
    trace.agent.name = "solver"
    calls = EpisodeCalls.continuing(saved_trace(trace))
    assert calls.start("judge", task("y"), None) == (("judge", 0), None)
    key, saved = calls.start("solver", task("anything"), None)
    assert (
        key == ("solver", 0)
        and saved["state"] == "running"
        and saved["trace"]["id"] == trace.id
    )


@pytest.mark.asyncio
async def test_a_diverged_attempt_starts_over_and_keeps_its_retries():
    seen: list[bool] = []

    class FakeEnv:
        config = vf.EnvConfig()

        async def run_episode(self, task, ctx, *, episode_id, attempt, calls, **_):
            seen.append(bool(calls.memo))
            if calls.memo:
                calls.diverged = "a call 0 was asked for another task than at the save"
            episode = vf.Episode(
                id=episode_id,
                attempt=attempt,
                task=new_trace("x").task,
                traces=[new_trace("x")],
            )
            episode.ok = True
            return episode

    point = {
        "episode_point": 1,
        "calls": [
            {
                "seat": "a",
                "n": 0,
                "fingerprint": "f",
                "parent": None,
                "state": "running",
                "trace": {},
            }
        ],
    }
    slot = RunSlot(task("x"), restore=point)
    episode = await Env.run_slot(FakeEnv(), slot, ctx=None)
    assert seen == [True, False]
    assert episode.attempt == 1 and episode.ok
