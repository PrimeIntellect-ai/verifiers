"""Borrowed boxes (`verifiers.v1.box`): a box from `agent.provision` is checkpointed
through its owner for the runs that borrow it, restored by its owner for an interaction
retry (every seat on it sees the restored box).

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
