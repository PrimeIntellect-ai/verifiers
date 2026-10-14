"""Borrowed boxes (`verifiers.v1.box`): a box from `agent.provision` is checkpointed
through its owner for the runs that borrow it, restored by its owner for an interaction
retry (every seat on it sees the restored box), and, with `key=`, restored after a
crash from the latest checkpoint its runs recorded so their durable calls resume on it.

Deterministic, no API key: the fakes of `test_agent_resume`.
"""

# ruff: noqa: F811 - the fixtures imported from test_agent_resume are parameters

import pytest
from test_agent_resume import (  # noqa: F401 - fixtures
    EVENTS,
    FakeRuntime,
    RestoreConfig,
    box_lost,
    fake_boxes,
    make,
    model,
    roles,
    task,
)

import verifiers.v1 as vf
import verifiers.v1.box as box_module
import verifiers.v1.runtimes as runtimes_module
from verifiers.v1.utils.retries import RETRY_NOTE


@pytest.fixture(autouse=True)
def owned_fake_boxes(monkeypatch):
    """`agent.provision` and the owner's restore make fake boxes too."""

    def fake(config, name=None):
        return FakeRuntime(config, name)

    def restore(config, checkpoint):
        return RestoreConfig(**config.model_dump(), checkpoint=checkpoint)

    monkeypatch.setattr(runtimes_module, "make_runtime", fake)
    monkeypatch.setattr(box_module, "make_runtime", fake)
    monkeypatch.setattr(box_module, "restore_config", restore)


@pytest.fixture
def store(tmp_path) -> vf.TraceDir:
    return vf.TraceDir(tmp_path / "traces")


async def test_a_borrowed_run_saves_per_call_through_the_owner(model):
    saves: list[vf.Checkpoint] = []
    agent = make(model, save="per_call")
    async with agent.provision(task()) as box:
        owner = vf.box_of(box)
        assert owner is not None and owner.save
        trace = await make(model).run(
            task(), runtime=box, on_save=lambda t, c: saves.append(c)
        )
        assert trace.ok, trace.errors
        assert trace.agent.runtime.borrowed
        assert [(c.id, c.turns) for c in trace.checkpoints] == [
            ("ckpt-1", 0),
            ("ckpt-2", 1),
            ("ckpt-3", 3),
        ]
        assert saves == trace.checkpoints
        assert owner.latest.id == "ckpt-3"
        # The box is the owner's: the run left it running.
        assert not box.stopped


async def test_a_borrowed_run_saves_when_it_saves_itself(model):
    async with make(model).provision(task()) as box:
        assert not vf.box_of(box).save
        plain = await make(model).run(task(), runtime=box)
        assert plain.checkpoints == []
        saving = await make(model, save="per_call").run(task(), runtime=box)
        assert len(saving.checkpoints) == 3


async def test_an_interaction_on_a_borrowed_box_checkpoints_through_the_owner(model):
    agent = make(model)
    async with (
        agent.provision(task()) as box,
        agent.interaction(task(None), runtime=box) as interaction,
    ):
        await interaction.turn("first")
        checkpoint = await interaction.checkpoint()
    assert checkpoint.id == "ckpt-1"
    assert vf.box_of(box).latest.id == "ckpt-1"


async def test_an_interaction_retry_has_the_owner_restore_the_box_for_every_seat(
    model,
):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.provision(task()) as box:
        view = box.with_env({"SEAT": "reviewer"})  # another seat's view of the box
        first_workdir = box.workdir
        async with agent.interaction(task(), runtime=box) as interaction:
            segment = await interaction.turn()
        assert segment.last_reply == f"done: {RETRY_NOTE}"
        (failed,) = interaction.failed_attempts
        assert not failed.ok
        retried = interaction.trace
        assert retried.ok, retried.errors
        # The owner restored the box from the latest checkpoint, in place: every
        # holder of the box (and every view) now acts on the restored one.
        assert box.source == "ckpt-2" and view.source == "ckpt-2"
        assert FakeRuntime.made[-1].source == "ckpt-2"
        assert box.workdir != first_workdir
        assert not first_workdir.exists()  # the lost box was torn down
        assert vf.box_of(box).generation == 1
        assert retried.checkpoints[0].restored
        assert retried.checkpoints[0].id == "ckpt-2"
        assert box.files() == {"a.txt": b"a.txt"}
    assert box.stopped


async def test_a_borrowed_retry_needs_a_checkpoint(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, interaction_retries=2)
    async with (
        agent.provision(task()) as box,
        agent.interaction(task(), runtime=box) as interaction,
    ):
        await interaction.turn()
        assert interaction.failed_attempts == []
    assert not interaction.trace.ok


async def test_two_seats_resume_on_the_box_their_owner_restores(
    model, store, monkeypatch
):
    """A control agent and its reviewer share one durable box. The episode dies in the
    reviewer's run; run again, the box comes back from the reviewer's latest
    checkpoint, the control agent's finished call is kept, and the reviewer's resumes
    on the restored box."""
    control = make(model, save="per_call")
    reviewer = make(model, save="per_call")

    async def episode():
        async with control.provision(task(), key="box") as box:
            first = await control.run(task("control"), runtime=box, key="control")
            second = await reviewer.run(task("review"), runtime=box, key="review")
            return box, first, second

    writes = {"b.txt": 0}
    original = FakeRuntime.write

    async def write(self, path, data):
        if path == "b.txt":
            writes["b.txt"] += 1
            if writes["b.txt"] == 2:  # the reviewer's second write dies with the box
                raise box_lost()
        await original(self, path, data)

    monkeypatch.setattr(FakeRuntime, "write", write)
    with vf.use_durable(vf.Durable(store, match={"job": "j1"})):
        box, first, failed = await episode()
        assert first.ok and not failed.ok
        assert first.labels["box"]["key"] == "agent/box"
        assert "checkpoint" not in first.labels["box"]  # the box was fresh
        assert failed.labels["box"]["fingerprint"] == first.labels["box"]["fingerprint"]
        # The reviewer started on the box as the control agent's last save left it.
        assert failed.labels["box"]["checkpoint"]["id"] == first.checkpoints[-1].id
        assert failed.labels["call"]["key"] == "agent/review"
        assert failed.labels["job"] == "j1"
        latest = failed.checkpoints[-1]
        model.requests.clear()
        EVENTS.clear()
        box, kept, resumed = await episode()
    assert kept.kept and kept.id == first.id
    assert resumed.ok, resumed.errors
    assert resumed.resumed_from == failed.id
    # The box came back from the latest checkpoint either seat took.
    assert box.source == latest.id
    assert resumed.checkpoints[0].restored and resumed.checkpoints[0].id == latest.id
    assert model.requests[0]["messages"][-1]["content"] == RETRY_NOTE
    users = [m["content"] for m in model.requests[0]["messages"] if m["role"] == "user"]
    assert users[0] == "review"


async def test_a_durable_box_starts_fresh_when_its_inputs_change(model, store):
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        async with agent.provision(task(), key="box") as box:
            await agent.run(task(), runtime=box)
        async with agent.provision(task(), key="box") as box:
            assert box.source is not None
        async with agent.provision(task("other"), key="box") as box:
            assert box.source is None
        async with agent.provision(task(), key="box", depends="v2") as box:
            assert box.source is None
        async with agent.provision(task(), key="other-box") as box:
            assert box.source is None


async def test_a_durable_run_on_an_unrestored_box_runs_fresh(model, store):
    """A cut-off run whose box was not restored by its owner (an unkeyed provision)
    cannot continue: the box does not match its checkpoints."""
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        async with agent.provision(task(), durable=False) as box:
            failed = await agent.run(task(), runtime=box, key="solve")
        assert not failed.ok and failed.checkpoints
        async with agent.provision(task(), durable=False) as box:
            fresh = await agent.run(task(), runtime=box, key="solve")
    assert fresh.ok and fresh.resumed_from is None


async def test_a_box_restores_from_where_a_run_started_before_it_saved(
    model, store, monkeypatch
):
    """The run that moved last on a durable box died before it took a checkpoint of
    its own: the box comes back from the checkpoint it started on (its box label)."""
    control = make(model, save="per_call")
    reviewer = make(model, save="per_call")

    async def episode():
        async with control.provision(task(), key="box") as box:
            first = await control.run(task("control"), runtime=box, key="control")
            second = await reviewer.run(task("review"), runtime=box, key="review")
            return box, first, second

    with vf.use_durable(vf.Durable(store)):
        FakeRuntime.failing_checkpoints = set(range(4, 100))  # the reviewer's saves
        original = FakeRuntime.write
        writes = {"b.txt": 0}

        async def write(self, path, data):
            if path == "b.txt":
                writes["b.txt"] += 1
                if writes["b.txt"] == 2:
                    raise box_lost()
            await original(self, path, data)

        monkeypatch.setattr(FakeRuntime, "write", write)
        box, first, failed = await episode()
        assert first.ok and not failed.ok and failed.checkpoints == []
        assert failed.labels["box"]["checkpoint"]["id"] == first.checkpoints[-1].id
        FakeRuntime.failing_checkpoints = set()
        box, kept, rerun = await episode()
    assert kept.kept
    assert box.source == first.checkpoints[-1].id
    assert rerun.ok and rerun.resumed_from is None


async def test_a_continuation_on_a_borrowed_box_starts_from_its_latest_checkpoint(
    model,
):
    """A box restored from one checkpoint and moved on by a run: a continuation on it
    records where the box is (its latest checkpoint, with that checkpoint's time), not
    where it was first restored from."""
    agent = make(model, save="per_call")
    FakeRuntime.store["X"] = {}
    async with agent.provision(task(), checkpoint="X") as box:
        owner = vf.box_of(box)
        first = await agent.run(task(), runtime=box)
        assert first.ok, first.errors
        moved = owner.latest
        assert owner.restored.id == "X" and moved.id == first.checkpoints[-1].id
        second = await make(model).run(resume=first, note="again", runtime=box)
    start = second.checkpoints[0]
    assert start.restored and start.turns == 0
    assert (start.id, start.time) == (moved.id, moved.time)


async def test_a_continuation_that_never_saves_keeps_the_boxs_latest_checkpoint(
    model, store
):
    """After a crash, a durable box comes back from its latest checkpoint even when the
    last run on it, a continuation, failed before saving: its start does not stand in
    for a newer checkpoint than it is."""
    from verifiers.v1.durable import box_checkpoint

    agent = make(model, save="per_call")
    durable = vf.Durable(store, match={"job": "j"})
    with vf.use_durable(durable):
        async with agent.provision(task(), key="box") as box:
            await agent.run(task(), runtime=box, key="first")
    with vf.use_durable(durable):
        async with agent.provision(task(), key="box") as box:
            owner = vf.box_of(box)
            other = await agent.run(task(), runtime=box, key="other")
            assert other.ok, other.errors
            moved = owner.latest
            assert moved.id != owner.restored.id
            model.refuse.add(len(model.requests))
            failed = await make(model).run(
                resume=other, note="again", runtime=box, key="second"
            )
            assert not failed.ok
            assert box_checkpoint(durable, owner.ref).id == moved.id


async def test_a_keyed_box_from_a_checkpoint_comes_back_where_its_runs_left_it(
    model, store
):
    """`provision(checkpoint=, key=)` after a crash: the box starts from the latest
    checkpoint its runs recorded, not from the given one, so the cut-off run resumes on
    a box that matches its conversation."""
    agent = make(model, save="per_call")
    FakeRuntime.store["X"] = {}
    with vf.use_durable(vf.Durable(store)):
        async with agent.provision(task(), checkpoint="X", key="box") as box:
            model.refuse.add(len(model.requests) + 2)  # cut off at the third call
            failed = await agent.run(task(), runtime=box, key="solve")
        assert not failed.ok
        latest = failed.checkpoints[-1]
        async with agent.provision(task(), checkpoint="X", key="box") as box:
            assert box.source == latest.id
            again = await agent.run(task(), runtime=box, key="solve")
        assert again.ok and again.resumed_from == failed.id
        assert box.files() == FakeRuntime.store[latest.id]
        # Another starting checkpoint is another box.
        FakeRuntime.store["Y"] = {}
        async with agent.provision(task(), checkpoint="Y", key="box") as box:
            assert box.source == "Y"


async def test_a_box_from_a_checkpoint_continues_no_later_checkpoint(model, store):
    """A durable call on a box started from a given checkpoint does not continue a
    later checkpoint of its trace: the box holds none of that work."""
    agent = make(model, save="per_call")
    FakeRuntime.store["X"] = {}
    with vf.use_durable(vf.Durable(store)):
        async with agent.provision(task(), checkpoint="X") as box:
            model.refuse.add(len(model.requests) + 2)
            failed = await agent.run(task(), runtime=box, key="solve")
        assert not failed.ok and failed.checkpoints
        async with agent.provision(task(), checkpoint="X") as box:
            fresh = await agent.run(task(), runtime=box, key="solve")
    assert fresh.ok and fresh.resumed_from is None


async def test_a_borrowed_retry_whose_box_will_not_start_ends_the_turn(
    model, monkeypatch
):
    """The owner cannot start the restored box: the turn ends with that failure on
    the replacement's trace instead of raising out of `turn()`."""
    FakeRuntime.doomed["b.txt"] = box_lost
    original = FakeRuntime.start

    async def start(self):
        if self.source is not None:
            raise vf.SandboxError("restored box would not start")
        await original(self)

    monkeypatch.setattr(FakeRuntime, "start", start)
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.provision(task()) as box:
        async with agent.interaction(task(), runtime=box) as interaction:
            segment = await interaction.turn()
        assert segment.terminated
        (failed,) = interaction.failed_attempts
        assert failed.errors[-1].type == "SandboxError"
        trace = interaction.trace
        assert not trace.ok and trace.resumed_from == failed.id
        assert "would not start" in trace.errors[-1].message
        assert vf.box_of(box).generation == 0


class CountingDir(vf.TraceDir):
    """A trace directory counting its writes, per trace."""

    def __init__(self, path) -> None:
        super().__init__(path)
        self.writes: dict[str, int] = {}

    def write(self, trace: vf.Trace) -> None:
        self.writes[trace.id] = self.writes.get(trace.id, 0) + 1
        super().write(trace)


async def test_a_durable_call_on_a_durable_box_is_written_once_per_save(
    model, tmp_path
):
    store = CountingDir(tmp_path / "traces")
    agent = make(model, save="per_call")
    with vf.use_durable(vf.Durable(store)):
        async with agent.provision(task(), key="box") as box:
            ran = await agent.run(task(), runtime=box, key="solve")
            async with agent.interaction(task(None), runtime=box, key="talk") as talk:
                await talk.turn("hello")
    # One write per per-call save and one when it finished, not one per path.
    assert store.writes[ran.id] == len(ran.checkpoints) + 1
    assert store.writes[talk.trace.id] == len(talk.trace.checkpoints) + 1


async def test_a_model_failure_on_a_borrowed_box_continues_on_it(model):
    """`Rollout.recover` holds for a borrowed box too: a failed model call continues on
    the same box, which the owner neither restores nor replaces."""
    model.refuse.add(1)
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.provision(task()) as box:
        async with agent.interaction(task(), runtime=box) as interaction:
            segment = await interaction.turn()
        assert interaction.failed_attempts == [] and interaction.trace.ok
        assert segment.last_reply.startswith("done")
        assert vf.box_of(box).generation == 0
