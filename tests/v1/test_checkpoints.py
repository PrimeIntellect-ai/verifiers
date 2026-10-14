"""Filesystem checkpoints: `Interaction.checkpoint` records a restorable checkpoint on
the trace, `checkpoint=` starts a run's box from one, and a run continuing a trace on a
box started from a checkpoint runs `Task.restore` instead of `Task.setup`.

Deterministic, no API key: a scripted OpenAI-compatible endpoint answers every call,
an in-process harness writes a file per segment and calls it through the real
interception server, and a subprocess runtime gains fake filesystem checkpoints.
"""

from pathlib import Path
from typing import ClassVar

import pytest
from aiohttp import web
from openai import AsyncOpenAI
from pydantic import ValidationError

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
import verifiers.v1.box as box_module
import verifiers.v1.rollout as rollout_module
import verifiers.v1.runtimes as runtimes_module
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult, SubprocessConfig
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

EVENTS: list[str] = []
SEEN: list[list[str]] = []
"""The files each segment found in its box."""


class NotesTask(vf.Task):
    """Records whether its box was set up fresh or restored from a checkpoint."""

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        EVENTS.append("setup")

    async def restore(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        EVENTS.append("restore")


class WritingHarness(NullHarness):
    """Writes `segment-<n>.txt` into the box, then makes one model call."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        SEEN.append(sorted(p.name for p in Path(runtime.workdir).iterdir()))
        await runtime.write(f"segment-{trace.num_turns}.txt", b"written")
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        try:
            await client.chat.completions.create(
                model=ctx.model, messages=[{"role": "user", "content": "go"}]
            )
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


class RestoreConfig(SubprocessConfig):
    checkpoint: str | None = None


class FakeRuntime(SubprocessRuntime):
    """A subprocess box with in-memory filesystem checkpoints."""

    store: ClassVar[dict[str, dict[str, bytes]]] = {}
    refuse: ClassVar[bool] = False
    made: ClassVar[list["FakeRuntime"]] = []

    def __init__(self, config, name=None) -> None:
        self.source = getattr(config, "checkpoint", None)
        super().__init__(
            SubprocessConfig(**config.model_dump(exclude={"checkpoint"})), name
        )
        FakeRuntime.made.append(self)

    async def start(self) -> None:
        await super().start()
        assert self.workdir is not None
        for path, data in self.store.get(self.source, {}).items():
            (self.workdir / path).write_bytes(data)

    @property
    def restored(self) -> bool:
        return self.source is not None

    async def checkpoint(self) -> str:
        if self.refuse:
            raise vf.SandboxError("checkpoint refused", kind="io")
        assert self.workdir is not None
        checkpoint = f"ckpt-{len(self.store) + 1}"
        self.store[checkpoint] = {
            str(path.relative_to(self.workdir)): path.read_bytes()
            for path in Path(self.workdir).rglob("*")
            if path.is_file()
        }
        return checkpoint

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.store.pop(checkpoint_id, None)


@pytest.fixture(autouse=True)
def fake_boxes(monkeypatch):
    EVENTS.clear()
    SEEN.clear()
    FakeRuntime.store = {}
    FakeRuntime.refuse = False
    FakeRuntime.made = []
    monkeypatch.setattr(
        rollout_module,
        "make_runtime",
        lambda config, name=None: FakeRuntime(config, name),
    )
    monkeypatch.setattr(
        agent_module,
        "restore_config",
        lambda config, checkpoint: RestoreConfig(
            **config.model_dump(), checkpoint=checkpoint
        ),
    )
    monkeypatch.setenv("SCRIPTED_KEY", "scripted")


@pytest.fixture
async def model():
    async def handle(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": "scripted",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    await runner.cleanup()


def make(url: str) -> vf.Agent:
    agent = vf.Agent(
        vf.AgentConfig(
            model="scripted",
            client={"type": "eval", "base_url": url, "api_key_var": "SCRIPTED_KEY"},
            runtime={"type": "subprocess"},
            harness={"id": "null"},
        )
    )
    agent.harness = WritingHarness(agent.harness.config)
    return agent


def task() -> NotesTask:
    return NotesTask(vf.TaskData(idx=0, prompt=None))


async def test_an_interaction_checkpoint_is_recorded_with_its_turns(model):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn("first")
        checkpoint = await interaction.checkpoint()
        await interaction.turn("second")
    assert interaction.trace.ok, interaction.trace.errors
    assert (checkpoint.id, checkpoint.turns) == ("ckpt-1", 1)
    assert interaction.trace.checkpoints == [checkpoint]
    assert sorted(FakeRuntime.store["ckpt-1"]) == ["segment-0.txt"]
    record = vf.WireTrace.model_validate(interaction.trace.to_record())
    assert record.checkpoints == [checkpoint]
    assert checkpoint.time > 0
    # Every checkpoint is recorded with its time: one without is no record.
    with pytest.raises(ValidationError):
        vf.Checkpoint.model_validate({"id": "ckpt-1", "turns": 1})


async def test_a_continuation_from_a_checkpoint_restores_its_files_and_skips_setup(
    model,
):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn("first")
        checkpoint = await interaction.checkpoint()
    assert EVENTS == ["setup"]

    EVENTS.clear()
    trace = await agent.run(resume=interaction.trace, note="again")
    assert trace.ok, trace.errors
    assert EVENTS == ["restore"]
    assert FakeRuntime.made[-1].source == checkpoint.id
    assert SEEN[-1] == ["segment-0.txt"]


async def test_a_fresh_run_on_a_box_from_a_checkpoint_sets_its_task_up(
    model, monkeypatch
):
    """Whether a run restores its task is the run's, not the box's: a run that
    continues no trace sets its task up, on its own box from a checkpoint or on a
    provisioned one other runs share."""
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn("first")
        checkpoint = await interaction.checkpoint()

    EVENTS.clear()
    other = NotesTask(vf.TaskData(idx=1, prompt="a different task"))
    trace = await agent.run(other, checkpoint=checkpoint.id)
    assert trace.ok, trace.errors
    assert EVENTS == ["setup"]
    assert SEEN[-1] == ["segment-0.txt"]  # the box still holds the checkpoint's files

    EVENTS.clear()
    restore = agent_module.restore_config
    monkeypatch.setattr(box_module, "restore_config", restore)
    monkeypatch.setattr(runtimes_module, "make_runtime", rollout_module.make_runtime)
    async with agent.provision(task(), checkpoint=checkpoint.id) as box:
        trace = await agent.run(other, runtime=box)
        assert trace.ok, trace.errors
        await agent.run(resume=interaction.trace, note="again", runtime=box)
    assert EVENTS == ["setup", "restore"]


async def test_a_failed_checkpoint_leaves_no_record(model):
    agent = make(model)
    FakeRuntime.refuse = True
    async with agent.interaction(task()) as interaction:
        await interaction.turn("first")
        with pytest.raises(vf.SandboxError, match="refused"):
            await interaction.checkpoint()
    assert interaction.trace.checkpoints == []


async def test_a_borrowed_box_without_an_owner_is_not_checkpointed(model):
    agent = make(model)
    async with (
        vf.runtimes.provision_runtime(SubprocessConfig()) as box,
        agent.interaction(task(), runtime=box) as interaction,
    ):
        await interaction.turn("first")
        with pytest.raises(ValueError, match="borrowed box"):
            await interaction.checkpoint()
    assert interaction.trace.checkpoints == []


async def test_a_borrowed_box_cannot_also_start_from_a_checkpoint(model):
    agent = make(model)
    async with agent.provision(task()) as box:
        with pytest.raises(ValueError, match="borrowed runtime"):
            await agent.run(task(), runtime=box, checkpoint="ckpt-1")
