"""An episode's agent calls are keyed by seat and call number, and a continued episode
replays them from its restore point: a finished call returns its saved trace without
running, a running call continues from its saved trace, and a call that made no turn
runs fresh."""

import asyncio
import pickle
from collections import defaultdict
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

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
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    calls.minted(
        ("solver", 0),
        SimpleNamespace(id="t-s", num_turns=3, agent=SimpleNamespace(runtime=None)),
    )
    calls.minted(
        ("judge", 0),
        SimpleNamespace(id="t-j", num_turns=1, agent=SimpleNamespace(runtime=None)),
    )
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


def in_box(trace: vf.Trace, box: str) -> vf.Trace:
    trace.agent.runtime = TypeAdapter(vf.RuntimeInfo).validate_python(
        {"type": "prime", "id": box}
    )
    return trace


def test_a_run_in_scoring_continues_from_the_box_its_judge_saved():
    """An agentic judge runs in the solver's box during the solver's scoring: the
    judge's snapshot of that box continues both, the solver by scoring again."""
    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    solver = in_box(new_trace("solve"), "box-1")
    solver.stop("agent_completed")
    solver.timing.finalize.end = 1.0
    judge = in_box(new_trace("judge"), "box-1")
    calls.minted(("solver", 0), solver)
    calls.minted(("judge", 0), judge)

    point, reason = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
    )

    assert reason is None
    point = pickle.loads(pickle.dumps(point))
    by_seat = {c["seat"]: c for c in point["calls"]}
    assert by_seat["judge"]["parent"] == ["solver", 0]
    (checkpoint,) = by_seat["solver"]["trace"]["checkpoints"]
    assert checkpoint["snapshot"] == {"id": f"s-{judge.id}"}
    assert (
        checkpoint["session"]["stage"] == "scoring"
        and checkpoint["session"]["finalized"]
    )
    assert checkpoint["session"]["stop"][0] == "agent_completed"
    # One box: both calls continue from the same snapshot.
    assert point["snapshots"] == [{"id": f"s-{judge.id}"}]

    assert checkpoint["session"]["box"] == "own"

    # Isolated scoring: the judge runs in the grading box, which its snapshot restores.
    in_box(judge, "grading-box")
    point, reason = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
    )
    assert reason is None
    (solver_entry,) = [c for c in point["calls"] if c["seat"] == "solver"]
    (checkpoint,) = solver_entry["trace"]["checkpoints"]
    assert checkpoint["session"]["box"] == "grading"
    assert checkpoint["snapshot"] == {"id": f"s-{judge.id}"}


@pytest.mark.asyncio
async def test_a_call_that_starts_over_runs_the_calls_of_its_scoring_fresh(monkeypatch):
    """The solver's restore led nowhere and it starts over in a new box: its judge's
    saved call belongs to the box it no longer continues, so the judge runs fresh."""
    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge solve"), ("solver", 0), borrowed=True)
    calls.start("helper", task("help"), ("judge", 0))
    for key in [("solver", 0), ("judge", 0), ("helper", 0)]:
        calls.minted(key, in_box(new_trace(key[0]), "box-1"))
    saved = [saved_trace(calls.calls[k].trace) for k in [("judge", 0), ("helper", 0)]]
    point, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), saved)
    assert reason is None

    started: list[tuple[str, bool]] = []

    async def run(
        self,
        handle,
        task,
        *,
        on_trace=None,
        on_score=None,
        restore=None,
        on_retry=None,
        **_,
    ):
        started.append((self._name, restore is not None))
        if restore is not None and self._name == "solver":
            on_retry(False)  # its snapshot is gone: the solver starts over
        trace = new_trace(task.data.prompt_text)
        on_trace(trace)
        if on_score is not None:
            await on_score(trace, None)
        trace.ok = True
        return trace

    monkeypatch.setattr(Agent, "_run", run)
    calls = EpisodeCalls.continuing(pickle.loads(pickle.dumps(point)))
    await judged(agents(calls, [], "solver", "judge"))
    assert started == [("solver", True), ("judge", False)]
    assert calls.memo == {}


def finished_call(calls: EpisodeCalls, seat: str, prompt: str, parent=None):
    key, _ = calls.start(seat, task(prompt), parent)
    trace = new_trace(prompt)
    calls.minted(key, trace)
    calls.finished(key, trace)
    return key, trace


def test_a_replay_brings_back_nested_calls_at_any_depth_in_the_order_they_finished():
    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    # The helper (inside the judge's scoring) finishes before the judge.
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    finished_call(calls, "helper", "help", ("judge", 0))
    judge = new_trace("judge")
    calls.minted(("judge", 0), judge)
    calls.finished(("judge", 0), judge)
    solver = new_trace("solve")
    calls.minted(("solver", 0), solver)
    calls.finished(("solver", 0), solver)
    point, _ = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])

    calls = EpisodeCalls.continuing(pickle.loads(pickle.dumps(point)))
    key, saved = calls.start("solver", task("solve"), None)
    _, nested = calls.replay(key, saved, task("solve"))
    assert [t.task.data.prompt_text for t in nested] == ["help", "judge"]
    assert calls.memo == {} and calls.counts["helper"] == 1
    assert type(nested[0].state) is vf.State


def test_a_continued_call_waiting_for_its_turn_keeps_its_saved_trace():
    running = new_trace("x")
    calls = EpisodeCalls()
    calls.start("a", task("x"), None)
    calls.minted(("a", 0), running)
    point, _ = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(running)]
    )

    # Continued, but not yet minted: the call waits for its agent permit.
    calls = EpisodeCalls.continuing(point)
    calls.start("a", task("x"), None)
    again, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert reason is None and again["calls"] == point["calls"]
    assert again["snapshots"] == point["snapshots"]


@pytest.mark.asyncio
async def test_a_diverged_attempt_stops_its_calls_that_still_run():
    calls = EpisodeCalls()
    stopped = asyncio.Event()

    async def sibling():
        calls.start("b", task("y"), None)
        try:
            await asyncio.sleep(60)
        finally:
            stopped.set()

    running = asyncio.ensure_future(sibling())
    await asyncio.sleep(0)
    await calls.cancel_running()
    assert stopped.is_set() and running.cancelled()


def test_a_scoring_point_keeps_the_outcome_a_rewind_clears_and_needs_a_borrowed_box():
    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    solver = in_box(new_trace("solve"), "box-1")
    solver.stop("agent_completed")
    solver.is_timeout = True
    solver.root_reply = "#### 4"
    judge = in_box(new_trace("judge"), "box-1")
    calls.minted(("solver", 0), solver)
    calls.minted(("judge", 0), judge)
    point, _ = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
    )
    (entry,) = [c for c in point["calls"] if c["seat"] == "solver"]
    session = entry["trace"]["checkpoints"][0]["session"]
    assert session["timeout"] is True and session["root_reply"] == "#### 4"

    # A judge in a box of its own holds nothing of the solver's scoring.
    calls.calls[("judge", 0)].borrowed = False
    point, reason = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
    )
    assert point is None and "inside another run's scoring" in reason


def scoring_episode():
    """A solver in scoring whose judge runs in the solver's box, saved."""
    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    solver = in_box(new_trace("solve"), "box-1")
    judge = in_box(new_trace("judge"), "box-1")
    calls.minted(("solver", 0), solver)
    calls.minted(("judge", 0), judge)
    point, _ = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
    )
    return point


def test_a_scoring_retry_from_the_saved_snapshot_offers_its_nested_calls_again():
    calls = EpisodeCalls.continuing(scoring_episode())
    calls.start("solver", task("solve"), None)
    key, first = calls.start("judge", task("judge"), ("solver", 0))
    assert first is not None
    # The solver's scoring failed and retries from the same snapshot: so does the judge.
    calls.retry_nested(("solver", 0), from_saved=True)
    assert calls.start("judge", task("judge"), ("solver", 0)) == (key, first)
    # The solver starts over in a new box: the judge runs fresh.
    calls.retry_nested(("solver", 0), from_saved=False)
    assert calls.start("judge", task("judge"), ("solver", 0))[1] is None


def test_saved_calls_the_replay_has_not_reached_stay_in_a_later_save():
    point = scoring_episode()
    calls = EpisodeCalls.continuing(point)
    again, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert reason is None
    assert sorted((c["seat"], c["n"]) for c in again["calls"]) == [
        ("judge", 0),
        ("solver", 0),
    ]
    assert again["snapshots"] == point["snapshots"]


def test_a_trace_the_caller_hands_over_is_logged_but_not_replayed():
    calls = EpisodeCalls.continuing(scoring_episode())
    key, saved = calls.start("solver", task("solve"), None, replay=False)
    assert key == ("solver", 0) and saved is None and ("solver", 0) not in calls.memo


def test_an_isolated_scoring_point_keeps_the_state_grading_staged():
    from verifiers.v1 import grading

    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    solver = in_box(new_trace("solve"), "box-1")
    judge = in_box(new_trace("judge"), "grading-box")
    calls.minted(("solver", 0), solver)
    calls.minted(("judge", 0), judge)
    staged = solver.model_copy(deep=True)
    staged.state.artifacts = {"/app/answer.txt": b"42"}
    grader, original = task("solve"), task("solve")
    grader.verifier_staged = True
    grading._grading[id(solver)] = (staged, grader, original)
    try:
        point, _ = episode_restore_point(
            SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
        )
    finally:
        grading._grading.pop(id(solver))
    (entry,) = [c for c in point["calls"] if c["seat"] == "solver"]
    session = entry["trace"]["checkpoints"][0]["session"]
    assert session["box"] == "grading" and session["grading_state"]["artifacts"]
    # What staging set on the grader task, which a restored grading box gets back.
    assert session["grading_task"] == {"verifier_staged": True}


def test_a_scoring_retry_offers_again_the_nested_calls_not_reached_or_replayed():
    calls = EpisodeCalls()
    calls.start("solver", task("solve"), None)
    finished_call(calls, "helper", "help", ("solver", 0))
    calls.start("judge", task("judge"), ("solver", 0), borrowed=True)
    solver = in_box(new_trace("solve"), "box-1")
    judge = in_box(new_trace("judge"), "box-1")
    calls.minted(("solver", 0), solver)
    calls.minted(("judge", 0), judge)
    point, _ = episode_restore_point(
        SimpleNamespace(id="e1", calls=calls), [saved_trace(judge)]
    )

    calls = EpisodeCalls.continuing(point)
    calls.start("solver", task("solve"), None)
    # The scoring replays the finished helper, fails before the judge starts, and
    # retries from the saved snapshot: both are offered again.
    key, saved = calls.start("helper", task("help"), ("solver", 0))
    calls.replay(key, saved, task("help"))
    calls.retry_nested(("solver", 0), from_saved=True)
    assert sorted(calls.memo) == [("helper", 0), ("judge", 0)]


def test_a_scoring_retry_keeps_the_keys_other_calls_hold():
    calls = EpisodeCalls.continuing(scoring_episode())
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0))
    # Another solver's judge took the next judge key meanwhile.
    calls.start("judge", task("other"), ("solver", 1))
    calls.retry_nested(("solver", 0), from_saved=True)
    # The judge takes its saved point again; the other judge keeps its key.
    assert calls.start("judge", task("judge"), ("solver", 0))[0] == ("judge", 0)
    assert ("judge", 1) in calls.calls
    assert calls.start("judge", task("next"), None)[0] == ("judge", 2)


def test_a_save_while_a_continued_run_restores_its_scoring_box_keeps_the_point():
    """The solver's scoring-stage trace is minted while its box restores, before its
    judge took its saved point: a save then still holds both."""
    point = scoring_episode()
    calls = EpisodeCalls.continuing(point)
    key, saved = calls.start("solver", task("solve"), None)
    calls.minted(key, vf.Trace.model_validate(saved["trace"]))
    again, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert reason is None
    assert sorted((c["seat"], c["n"]) for c in again["calls"]) == [
        ("judge", 0),
        ("solver", 0),
    ]
    assert again["snapshots"] == point["snapshots"]


def test_a_grading_retry_offers_the_nested_saved_calls_again():
    calls = EpisodeCalls.continuing(scoring_episode())
    seats = agents(calls, [], "solver", "judge")
    calls.start("solver", task("solve"), None)

    async def judge(solution, runtime):
        pass

    score = seats["solver"]._scoring_callback(judge, ("solver", 0))
    key, saved = calls.start("judge", task("judge"), ("solver", 0))
    assert saved is not None and key not in calls.memo
    score.retry_saved()
    assert calls.memo[key] == saved


def test_a_new_call_skips_keys_a_replayed_call_took():
    calls = EpisodeCalls()
    calls.calls[("judge", 0)] = SimpleNamespace(parent=None)
    assert calls.start("judge", task("x"), None)[0] == ("judge", 1)


def test_a_save_while_a_nested_call_waits_for_its_permit_keeps_the_point():
    point = scoring_episode()
    calls = EpisodeCalls.continuing(point)
    key, saved = calls.start("solver", task("solve"), None)
    calls.minted(key, vf.Trace.model_validate(saved["trace"]))
    # The judge took its saved point but waits for its agent permit.
    calls.start("judge", task("judge"), ("solver", 0))
    again, reason = episode_restore_point(SimpleNamespace(id="e1", calls=calls), [])
    assert reason is None
    assert sorted((c["seat"], c["n"]) for c in again["calls"]) == [
        ("judge", 0),
        ("solver", 0),
    ]


def test_a_saved_call_of_another_calls_scoring_keeps_its_key():
    """Solver 0's scoring retries and rewinds the judge counter. Meanwhile solver 1's
    judge starts: it skips the key of solver 0's saved judge, which still takes it."""
    calls = EpisodeCalls.continuing(scoring_episode())
    calls.start("solver", task("solve"), None)
    calls.start("judge", task("judge"), ("solver", 0))
    calls.retry_nested(("solver", 0), from_saved=True)
    other, saved = calls.start("judge", task("other"), ("solver", 1))
    assert other == ("judge", 1) and saved is None
    key, saved = calls.start("judge", task("judge"), ("solver", 0))
    assert key == ("judge", 0) and saved is not None
