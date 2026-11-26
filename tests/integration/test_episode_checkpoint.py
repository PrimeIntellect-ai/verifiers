"""A multi-agent episode continues from a restore point another process saved.

An owner saves the running rollouts (`save_all`) and builds the episode's restore point
from them (`episode_restore_point`). A fresh env continues the episode
(`RunSlot.restore`): it runs the env's program again from the start, and each agent call,
keyed by its seat and its number within the episode, replays. A call that had finished
returns its saved trace without running. A call that was running continues from its
memory snapshot. A call that had not started runs fresh.

Every case runs its episode twice. The first run is not saved and is the reference. The
second run is saved while one chosen model call is open, and a fresh env continues it.
Scripted models answer every seat, so the continued episode must end with the reference's
messages and rewards. The models must also see no call for a turn that the restore point
already holds: a finished call that runs again would show here.

A case names the phase that makes it restorable (`SUPPORTED`). Until then, the case
checks the behavior of today: no restore point, the reason why, and an episode that
starts over and still ends like the reference.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pytest
from conftest import _eval_config
from test_checkpoint import assert_no_leaks, taken  # noqa: F401 - `taken` is a fixture
from tick_restore import COMMAND, LOG, TURNS

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.env import RunSlot
from verifiers.v1.envs.agentic_judge.env import VERDICT_FILE
from verifiers.v1.envs.user_sim.env import DONE
from verifiers.v1.restore import episode_restore_point, restore_entry, save_all
from verifiers.v1.runtimes import PrimeBackend, Snapshot
from verifiers.v1.utils.loaders import load_environment

SUPPORTED = 2
"""The highest phase this branch implements:
1. Agent runs in boxes of their own: fan-out and pipelines (`tick-duet`, `tick-pipeline`),
   and an agentic judge's solver before its scoring starts.
2. An agent run inside another run's scoring, in that run's box: the agentic judge with
   shared scoring. The judge's snapshot of the box continues both: the solver scores
   again in the restored box, and the judge continues in it.
3. The same with isolated scoring, where the judge runs in a scoring box of its own.
4. Interactions: the turns of an `interaction()` replay, and the open turn continues
   from its snapshot (user-sim)."""

# Each step is a tool call ("bash", {...}) or a text reply ("say", "...").
TICK = [("bash", {"command": COMMAND})] * TURNS
VERDICT = json.dumps(
    {"verdicts": [{"name": "solved", "reason": "1..4 in order", "verdict": "yes"}]}
)
JUDGE = [
    # The first command sleeps like a tick, so a save in that turn finds it running.
    ("bash", {"command": f"sleep 3; cat {LOG}"}),
    ("bash", {"command": f"printf '%s' '{VERDICT}' > {VERDICT_FILE}"}),
]
ASSISTANT = [
    ("bash", {"command": COMMAND}),
    ("say", "I ran the command."),
    ("bash", {"command": COMMAND}),
    ("say", "I ran it again."),
]
USER = [
    ("say", "Please run the tick command."),
    ("say", "Run it once more."),
    ("say", DONE),
]


class ScriptedModels:
    """A chat-completions server that answers each seat from its own script, chosen by
    the request's model name. A conversation's turn is the number of assistant messages
    it holds plus one, so a restored box that asks again gets the same reply. A script
    that has run out ends the conversation with a final text reply.

    `hold` keeps back the reply of one turn until `release`: the rollout is then known
    to be in that model call when the test saves it. `calls` counts the requests as they
    arrive, so a held request counts once, in the run that sent it."""

    def __init__(self) -> None:
        self.scripts: dict[str, list[tuple[str, object]]] = {}
        self.calls: Counter[tuple[str, int]] = Counter()
        self.arrived: defaultdict[tuple[str, int], asyncio.Event] = defaultdict(
            asyncio.Event
        )
        self.gates: dict[tuple[str, int], asyncio.Event] = {}
        self.base_url = ""

    def hold(self, model: str, turn: int) -> None:
        self.gates[(model, turn)] = asyncio.Event()

    async def held(self, model: str, turn: int) -> None:
        await self.arrived[(model, turn)].wait()

    def release(self) -> None:
        for gate in self.gates.values():
            gate.set()

    def reset(self) -> None:
        self.release()
        self.calls.clear()
        self.arrived.clear()
        self.gates.clear()

    def turns(self, model: str) -> list[int]:
        """The turns the model was asked for, with repeats."""
        return sorted(
            turn
            for (name, turn), n in self.calls.items()
            if name == model
            for _ in range(n)
        )

    async def answer(self, request):
        from aiohttp import web

        body = await request.json()
        model = str(body.get("model"))
        turn = sum(m.get("role") == "assistant" for m in body["messages"]) + 1
        self.calls[(model, turn)] += 1
        self.arrived[(model, turn)].set()
        if (gate := self.gates.get((model, turn))) is not None:
            await gate.wait()
        script = self.scripts[model]
        if turn <= len(script) and script[turn - 1][0] != "say":
            name, arguments = script[turn - 1]
            call = {
                "id": f"call_{turn}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            message, finish = (
                {"role": "assistant", "content": None, "tool_calls": [call]},
                "tool_calls",
            )
        else:
            text = script[turn - 1][1] if turn <= len(script) else "done"
            message, finish = {"role": "assistant", "content": text}, "stop"
        raw = {
            "id": f"chatcmpl-{model}-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        dialect = ChatDialect()
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response


@pytest.fixture
async def models():
    from aiohttp import web

    server = ScriptedModels()
    app = web.Application()
    app.router.add_post("/{tail:.*}", server.answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server.base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    try:
        yield server
    finally:
        server.release()
        await runner.cleanup()


@dataclass
class Case:
    name: str
    taskset: str
    env: dict
    scripts: dict[str, list]
    at: list[tuple[str, int]]
    """The model calls held open when the test saves: (seat model, turn)."""
    finished: list[str] = field(default_factory=list)
    """Seats whose run must have finished before the save."""
    phase: int = 1
    restored: dict[str, int | None] = field(default_factory=dict)
    """Once restorable: per seat model, the first turn the models see again after the
    continue (the open call is asked again), or None for a seat whose turns all replay."""
    skip_reason: str = ""
    """Before its phase: the reason the episode has no restore point."""


def seat(model: str, harness: str = "bash") -> dict:
    return {"model": model, "harness": {"id": harness}, "runtime": {"type": "prime"}}


JUDGE_ENV = {
    "id": "agentic-judge",
    "solver": seat("solver"),
    "judge": seat("judge"),
    "task": {"prompt": f"Check that {LOG} holds 1..{TURNS}, one number per line."},
    "score": {"task_weight": 0.5},
}
JUDGE_SCRIPTS = {"solver": TICK, "judge": JUDGE}
LAST_TICK = TURNS + 1  # the solver's reply after its last command

CASES = [
    # Agentic judge, shared scoring: the judge runs in the solver's box.
    Case(
        "judge-shared/solver-turn-2",
        "tick-restore",
        JUDGE_ENV,
        JUDGE_SCRIPTS,
        [("solver", 2)],
        restored={"solver": 2, "judge": 1},
    ),
    Case(
        "judge-shared/solver-last-turn",
        "tick-restore",
        JUDGE_ENV,
        JUDGE_SCRIPTS,
        [("solver", LAST_TICK)],
        restored={"solver": LAST_TICK, "judge": 1},
    ),
    Case(
        "judge-shared/judge-turn-1",
        "tick-restore",
        JUDGE_ENV,
        JUDGE_SCRIPTS,
        [("judge", 1)],
        phase=2,
        restored={"solver": None, "judge": 1},
        skip_reason="it runs inside another run's scoring",
    ),
    Case(
        "judge-shared/judge-turn-3",
        "tick-restore",
        JUDGE_ENV,
        JUDGE_SCRIPTS,
        [("judge", 3)],
        phase=2,
        restored={"solver": None, "judge": 3},
        skip_reason="it runs inside another run's scoring",
    ),
    # Agentic judge, isolated scoring: the judge runs in a scoring box of its own.
    Case(
        "judge-isolated/judge-turn-2",
        "tick-restore",
        {**JUDGE_ENV, "solver": {**seat("solver"), "scoring": {"mode": "isolated"}}},
        JUDGE_SCRIPTS,
        [("judge", 2)],
        phase=3,
        restored={"solver": None, "judge": 2},
        skip_reason="it runs inside another run's scoring",
    ),
    # Fan-out: both seats run at once.
    Case(
        "duet/both-running",
        "tick-duet",
        # Both seats run at once only with two agent permits.
        {"a": seat("a"), "b": seat("b"), "max_concurrent_agents": 2},
        {"a": TICK, "b": TICK},
        [("a", 2), ("b", 3)],
        restored={"a": 2, "b": 3},
    ),
    Case(
        "duet/one-finished",
        "tick-duet",
        {"a": seat("a"), "b": seat("b")},
        {"a": TICK, "b": TICK},
        [("b", 2)],
        finished=["a"],
        restored={"a": None, "b": 2},
    ),
    # Pipeline: the second seat's task is built from the first seat's trace.
    Case(
        "pipeline/first-running",
        "tick-pipeline",
        {"a": seat("a"), "b": seat("b")},
        {"a": TICK, "b": TICK},
        [("a", 2)],
        restored={"a": 2, "b": 1},
    ),
    Case(
        "pipeline/second-running",
        "tick-pipeline",
        {"a": seat("a"), "b": seat("b")},
        {"a": TICK, "b": TICK},
        [("b", 2)],
        restored={"a": None, "b": 2},
    ),
    # Interactions: the user (a box-less chat loop) and the assistant talk turn by turn.
    Case(
        "user-sim/assistant-turn",
        "echo",
        {
            "id": "user-sim",
            "user": seat("user", "null"),
            "assistant": seat("assistant"),
        },
        {"user": USER, "assistant": ASSISTANT},
        [("assistant", 1)],
        phase=4,
        restored={"user": 2, "assistant": 1},
        skip_reason="it is an interaction, which no checkpoint can continue",
    ),
    Case(
        "user-sim/user-turn",
        "echo",
        {
            "id": "user-sim",
            "user": seat("user", "null"),
            "assistant": seat("assistant"),
        },
        {"user": USER, "assistant": ASSISTANT},
        [("user", 2)],
        phase=4,
        restored={"user": 2, "assistant": 3},
        skip_reason="it is an interaction, which no checkpoint can continue",
    ),
]


def config_for(case: Case, models: ScriptedModels, **env):
    models.scripts = case.scripts
    return _eval_config(
        case.taskset,
        harness=None,
        env={**case.env, **env},
        max_turns=12,
        max_tokens=4096,
        rollout_timeout=600,
        client={"base_url": models.base_url},
    )


def context(config) -> ModelContext:
    return ModelContext(
        client=config.client, model=config.model, sampling=config.sampling
    )


def messages(trace) -> list[dict]:
    return [node.message.model_dump(exclude_none=True) for node in trace.nodes]


def by_seat(episode) -> dict[str, object]:
    return {trace.agent.name: trace for trace in episode.traces}


def finished(trace) -> bool:
    return trace.timing.scoring.end is not None


async def run_episode(
    config, restore: dict | None = None, episode_id: str | None = None
):
    env = load_environment(config.env)
    task = next(iter(env.taskset))
    async with env.serving():
        slot = (
            RunSlot(task, restore=restore)
            if episode_id is None
            else RunSlot(task, id=episode_id, restore=restore)
        )
        return await env.run_slot(slot, context(config))


async def saved_episode(case: Case, config, models: ScriptedModels):
    """Run the episode until every call in `case.at` is open and every seat in
    `case.finished` is done, save it, and cancel it with everything it holds. Returns
    the slot, the episode's restore point (a pickle round trip, as an owner stores it),
    and why there is none."""
    env = load_environment(config.env)
    task = next(iter(env.taskset))
    for model, turn in case.at:
        models.hold(model, turn)
    async with env.serving():
        (slot,) = env.slots(task)
        running = asyncio.ensure_future(env.run_slot(slot, context(config)))
        try:
            async with asyncio.timeout(600):
                for model, turn in case.at:
                    await models.held(model, turn)
                while not all(
                    any(t.agent.name == name and finished(t) for t in slot.traces)
                    for name in case.finished
                ):
                    await asyncio.sleep(0.2)
            points = await save_all()
            point, reason = episode_restore_point(slot, points)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            models.release()
    # The owner deletes what it saved but the point does not continue from, once the
    # boxes are gone: Prime refuses the newest snapshot of a box that still runs.
    kept = {s["id"] for s in point["snapshots"]} if point is not None else set()
    unused = [restore_entry(p)["snapshot"] for p in points]
    unused = [Snapshot.model_validate(s) for s in unused if s["id"] not in kept]
    if unused:
        assert len(await PrimeBackend.discard(unused)) == len(unused)
    return (
        slot,
        (pickle.loads(pickle.dumps(point)) if point is not None else None),
        reason,
    )


async def discard_point(point: dict) -> None:
    """The owner deletes the snapshots of a restore point once nothing restores it."""
    snapshots = [Snapshot.model_validate(s) for s in point["snapshots"]]
    deleted = await PrimeBackend.discard(snapshots)
    assert sorted(deleted) == sorted(s.id for s in snapshots)


def assert_same_episode(continued, reference) -> None:
    assert continued.ok, [e.message for t in continued.traces for e in t.errors]
    got, want = by_seat(continued), by_seat(reference)
    assert sorted(got) == sorted(want)
    for name, trace in want.items():
        assert messages(got[name]) == messages(trace), name
        assert got[name].reward == trace.reward, name


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_a_saved_episode_continues_to_the_reference(case, models, taken):  # noqa: F811
    config = config_for(case, models)
    reference = await run_episode(config)
    assert reference.ok, [e.message for t in reference.traces for e in t.errors]
    models.reset()

    slot, point, reason = await saved_episode(case, config, models)
    models.reset()
    if case.phase > SUPPORTED:
        assert point is None and case.skip_reason in (reason or "")
        # Nothing restores it: the episode starts over, every seat from its first turn.
        continued = await run_episode(config)
        assert_same_episode(continued, reference)
        for model in case.scripts:
            assert models.turns(model)[0] == 1, model
        await assert_no_leaks(taken)
        return

    assert point is not None, reason
    continued = await run_episode(config, restore=point, episode_id=slot.id)
    assert_same_episode(continued, reference)
    for model, first in case.restored.items():
        turns = models.turns(model)
        if first is None:
            # Every turn replays from the restore point: the model sees none.
            assert turns == [], model
        else:
            # Turns before the open call replay; the open call is asked once more.
            assert turns == list(range(first, turns[-1] + 1)), model
    assert any(t.has_restored for t in continued.traces)
    await discard_point(point)
    await assert_no_leaks(taken)


async def test_a_replay_that_diverges_starts_the_episode_over(models, taken):  # noqa: F811
    """The second seat's task changed between the save and the continue: the replay
    stops at that call, and the episode starts over with every seat from its first
    turn."""
    (case,) = [c for c in CASES if c.name == "pipeline/second-running"]
    config = config_for(case, models)
    slot, point, _ = await saved_episode(case, config, models)
    assert point is not None
    models.reset()
    changed = config_for(case, models, note="This line was not there at the save.")
    continued = await run_episode(changed, restore=point, episode_id=slot.id)
    assert continued.ok, [e.message for t in continued.traces for e in t.errors]
    assert models.turns("a")[0] == 1 and models.turns("b")[0] == 1
    assert not any(t.has_restored for t in continued.traces)
    await discard_point(point)
    await assert_no_leaks(taken)


async def test_a_lost_snapshot_starts_only_its_call_over(models, taken):  # noqa: F811
    """The snapshot of the running call is gone before the continue: that call starts
    over in a new box, and the finished call still replays without running."""
    (case,) = [c for c in CASES if c.name == "pipeline/second-running"]
    config = config_for(case, models)
    reference = await run_episode(config)
    models.reset()
    slot, point, _ = await saved_episode(case, config, models)
    assert point is not None
    await discard_point(point)
    models.reset()
    continued = await run_episode(config, restore=point, episode_id=slot.id)
    assert_same_episode(continued, reference)
    assert models.turns("a") == []
    assert models.turns("b")[0] == 1
    await assert_no_leaks(taken)
