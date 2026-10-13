"""Resuming from a trace: `Interaction.checkpoint`, `resume=` continuations (a trace
record, round-tripped through JSON), per-call saves (`AgentConfig.save`) and
interaction retries (`AgentConfig.interaction_retries`), all on `Trace.checkpoints`.

Deterministic, no API key: a scripted OpenAI-compatible endpoint answers from the
conversation alone, an in-process harness drives it through the real interception
server, and a subprocess runtime gains fake filesystem checkpoints.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest
from aiohttp import web
from openai import AsyncOpenAI

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
import verifiers.v1.rollout as rollout_module
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult, SubprocessConfig
from verifiers.v1.runtimes.subprocess import SubprocessRuntime
from verifiers.v1.utils.retries import RETRY_NOTE, retryable_error

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            },
        },
    }
]

EVENTS: list[str] = []


class NotesTask(vf.Task):
    """Records whether its box was set up fresh or restored from a checkpoint."""

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        EVENTS.append("setup")

    async def restore(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        EVENTS.append("restore")


def reply(messages: list[dict]) -> dict:
    """The scripted model: write a.txt, then b.txt, then answer the last user turn."""
    tools_seen = sum(message["role"] == "tool" for message in messages)
    if tools_seen < 2:
        name = "ab"[tools_seen] + ".txt"
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{len(messages)}",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": name, "text": name}),
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        users = [m["content"] for m in messages if m["role"] == "user"]
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


class ScriptHarness(NullHarness):
    """An in-process tool loop: model calls go through the interception endpoint,
    the `write` tool writes into the runtime."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        system, prompt = self.resolve_prompt(data)
        messages: list[dict] = [{"role": "system", "content": system}] if system else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages += [message_to_wire(message) for message in prompt or []]
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        try:
            while True:
                completion = await client.chat.completions.create(
                    model=ctx.model, messages=messages, tools=TOOLS
                )
                message = completion.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    return ProgramResult(exit_code=0, stdout="", stderr="")
                for call in message.tool_calls:
                    args = json.loads(call.function.arguments)
                    await runtime.write(args["path"], args["text"].encode())
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": "ok"}
                    )
        finally:
            await client.close()


class NoResumeHarness(ScriptHarness):
    SUPPORTS_RESUME = False


class RestoreConfig(SubprocessConfig):
    checkpoint: str | None = None


class FakeRuntime(SubprocessRuntime):
    """A subprocess box with in-memory filesystem checkpoints."""

    store: ClassVar[dict[str, dict[str, bytes]]] = {}
    failing_checkpoints: ClassVar[set[int]] = set()
    """Checkpoint attempts (1-based, across boxes) that fail."""
    attempts: ClassVar[int] = 0
    doomed: ClassVar[dict[str, Callable[[], Exception]]] = {}
    """Paths whose next write raises the mapped error once."""
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

    async def write(self, path: str, data: bytes) -> None:
        if (error := self.doomed.pop(path, None)) is not None:
            raise error()
        await super().write(path, data)

    async def checkpoint(self) -> str:
        FakeRuntime.attempts += 1
        if FakeRuntime.attempts in self.failing_checkpoints:
            raise vf.SandboxError("checkpoint refused")
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

    def files(self) -> dict[str, bytes]:
        return dict(self.store[self.source]) if self.source else {}


def box_lost() -> Exception:
    error = vf.SandboxError("sandbox terminated")
    error.kind = "box_lost"  # the typed kind, before `SandboxError(kind=...)` lands
    return error


@pytest.fixture(autouse=True)
def fake_boxes(monkeypatch):
    EVENTS.clear()
    FakeRuntime.store = {}
    FakeRuntime.failing_checkpoints = set()
    FakeRuntime.attempts = 0
    FakeRuntime.doomed = {}
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


@dataclass
class Endpoint:
    url: str
    requests: list[dict] = field(default_factory=list)
    refuse: set[int] = field(default_factory=set)
    """Indexes of requests the endpoint refuses (HTTP 400), as a provider would."""


@pytest.fixture
async def model():
    """The scripted endpoint, recording the requests it serves."""
    requests: list[dict] = []
    refuse: set[int] = set()

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        if len(requests) - 1 in refuse:
            return web.json_response({"error": {"message": "refused"}}, status=400)
        return web.json_response(reply(body["messages"]))

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield Endpoint(f"http://127.0.0.1:{port}/v1", requests, refuse)
    await runner.cleanup()


def make(model: Endpoint, harness=ScriptHarness, **config) -> vf.Agent:
    agent = vf.Agent(
        vf.AgentConfig(
            model="scripted",
            client={
                "type": "eval",
                "base_url": model.url,
                "api_key_var": "SCRIPTED_KEY",
            },
            runtime={"type": "subprocess"},
            harness={"id": "null"},
            **config,
        )
    )
    agent.harness = harness(agent.harness.config)
    return agent


def task(prompt: str | None = "start") -> NotesTask:
    return NotesTask(vf.TaskData(idx=0, prompt=prompt))


def roles(messages) -> list[str]:
    return [message["role"] for message in messages]


def reload(trace: vf.Trace) -> vf.WireTrace:
    """The trace as a consumer holds it: its JSON record, loaded back."""
    return vf.WireTrace.model_validate(json.loads(json.dumps(trace.to_record())))


# -- default path ---------------------------------------------------------------


async def test_default_path_adds_no_save_hook(model):
    agent = make(model)
    trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert trace.checkpoints == []
    assert FakeRuntime.attempts == 0
    assert len(model.requests) == 3
    assert trace.agent.config.save == "off"
    assert trace.agent.config.interaction_retries == 0
    # No save hook means no response interceptor: the server's rewrite path stays off.
    params = agent._rollout_params(task(), None, {})
    run = rollout_module.Rollout(task=task(), **params)
    assert run._session.response_interceptors == []


async def test_default_interaction_does_not_retry(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        assert interaction.failed_attempts == []
    assert not interaction.trace.ok
    assert interaction.trace.errors[-1].type == "SandboxError"
    assert len(FakeRuntime.made) == 1


# -- per-call saves -------------------------------------------------------------


async def test_per_call_saves_record_checkpoints_on_the_trace(model):
    saves: list[tuple[vf.Trace, vf.Checkpoint]] = []
    agent = make(model, save="per_call")
    trace = await agent.run(task(), on_save=lambda t, c: saves.append((t, c)))
    assert trace.ok, trace.errors
    assert all(seen is trace for seen, _ in saves)
    assert [checkpoint for _, checkpoint in saves] == trace.checkpoints
    assert [(c.id, c.turns) for c in trace.checkpoints] == [
        ("ckpt-1", 0),
        ("ckpt-2", 1),
        # A final reply leaves the box as it is: its checkpoint counts it.
        ("ckpt-3", 3),
    ]
    # Each save is taken before the harness runs the response's tool calls: the box
    # holds the files of earlier calls only, and the cut leaves the pending tool
    # call out.
    assert FakeRuntime.store["ckpt-1"] == {}
    assert FakeRuntime.store["ckpt-2"] == {"a.txt": b"a.txt"}
    assert FakeRuntime.store["ckpt-3"] == {"a.txt": b"a.txt", "b.txt": b"b.txt"}
    cuts = [trace.conversation_at(c.turns) for c in trace.checkpoints]
    assert [m.role for m in cuts[0]] == ["user"]
    assert [m.role for m in cuts[1]] == ["user", "assistant", "tool"]
    assert [m.role for m in cuts[2]][-1] == "assistant"
    assert cuts[2][-1].content == "done: start"
    assert trace.task.cls.endswith(":NotesTask")


async def test_a_refused_checkpoint_is_tried_again(model, monkeypatch):
    monkeypatch.setattr(rollout_module, "SAVE_RETRY_DELAY", 0.0)
    FakeRuntime.failing_checkpoints = {2}
    saves: list[vf.Checkpoint] = []
    agent = make(model, save="per_call")
    trace = await agent.run(task(), on_save=lambda t, c: saves.append(c))
    assert trace.ok, trace.errors
    assert [c.turns for c in saves] == [0, 1, 3]
    # The refused attempt's id never stays on the trace as a resume point.
    assert [c.id for c in trace.checkpoints] == ["ckpt-1", "ckpt-2", "ckpt-3"]


async def test_a_checkpoint_refused_every_time_is_skipped(model, monkeypatch):
    monkeypatch.setattr(rollout_module, "SAVE_RETRY_DELAY", 0.0)
    FakeRuntime.failing_checkpoints = {2, 3, 4}
    saves: list[vf.Checkpoint] = []
    agent = make(model, save="per_call")
    trace = await agent.run(task(), on_save=lambda t, c: saves.append(c))
    assert trace.ok, trace.errors
    assert [c.turns for c in saves] == [0, 3]
    assert [(c.id, c.turns) for c in trace.checkpoints] == [
        ("ckpt-1", 0),
        ("ckpt-2", 3),
    ]


async def test_a_trace_saved_in_on_save_resumes_from_what_the_box_holds(model):
    """on_save runs once the call has committed, so a trace persisted right there cuts
    at the new checkpoint to exactly the conversation the box matches."""
    saved: list[tuple[dict, vf.Checkpoint]] = []
    agent = make(model, save="per_call")
    await agent.run(
        task(), on_save=lambda t, c: saved.append((t.model_dump(mode="json"), c))
    )
    record, checkpoint = saved[1]
    trace = vf.WireTrace.model_validate(record)
    assert [m.role for m in trace.conversation_at(checkpoint.turns)] == [
        "user",
        "assistant",
        "tool",
    ]


async def test_on_save_failure_does_not_fail_the_run(model):
    def broken(trace: vf.Trace, checkpoint: vf.Checkpoint) -> None:
        raise OSError("disk full")

    trace = await make(model, save="per_call").run(task(), on_save=broken)
    assert trace.ok, trace.errors
    assert len(trace.checkpoints) == 3


# -- checkpoint and resume ------------------------------------------------------


async def test_checkpoint_resume_round_trip(model):
    agent = make(model)
    async with agent.interaction(task(prompt=None)) as interaction:
        segment = await interaction.turn("first")
        assert segment.last_reply == "done: first"
        checkpoint = await interaction.checkpoint()
        assert interaction.latest_checkpoint is checkpoint
    assert (checkpoint.id, checkpoint.turns) == ("ckpt-1", 3)
    loaded = reload(interaction.trace)
    assert loaded.checkpoints == [checkpoint]
    assert [m.role for m in loaded.conversation_at(checkpoint.turns)] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]

    EVENTS.clear()
    model.requests.clear()
    # The conversation ends on the agent's reply: the first turn continues it.
    async with agent.interaction(resume=loaded) as reopened:
        segment = await reopened.turn("second")
        assert segment.last_reply == "done: second"
    assert reopened.trace.ok, reopened.trace.errors
    assert EVENTS == ["restore"]
    assert FakeRuntime.made[-1].files() == {"a.txt": b"a.txt", "b.txt": b"b.txt"}
    # The prior turns reach the model as prompt context.
    assert roles(model.requests[0]["messages"]) == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert model.requests[0]["messages"][-1]["content"] == "second"
    assert reopened.trace.num_turns == 1
    assert type(reopened.task) is NotesTask
    # The box it started from is on the trace, matching its opening conversation.
    assert [(c.id, c.turns, c.restored) for c in reopened.trace.checkpoints] == [
        ("ckpt-1", 0, True)
    ]


async def test_a_resumed_trace_resumes_again_from_its_start(model):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        await interaction.checkpoint()
    first = await agent.run(resume=reload(interaction.trace), note="again")
    assert first.ok, first.errors
    model.requests.clear()
    # No checkpoint of its own: it continues from the box it started from.
    second = await agent.run(resume=reload(first), note="once more")
    assert second.ok, second.errors
    sent = model.requests[0]["messages"]
    assert [m["content"] for m in sent if m["role"] == "user"] == [
        "start",
        "again",
        "once more",
    ]
    assert FakeRuntime.made[-1].source == "ckpt-1"


async def test_resume_at_a_middle_checkpoint_gets_exactly_that_prefix(model):
    agent = make(model, save="per_call")
    trace = await agent.run(task())
    assert [c.id for c in trace.checkpoints] == ["ckpt-1", "ckpt-2", "ckpt-3"]
    loaded = reload(trace)
    for at in ("ckpt-2", 1):
        model.requests.clear()
        resumed = await make(model).run(resume=loaded, at=at)
        assert resumed.ok, resumed.errors
        # The cut is the request of the call the checkpoint was taken at: the first
        # write's result, without the second write.
        assert roles(model.requests[0]["messages"]) == ["user", "assistant", "tool"]
        assert FakeRuntime.made[-1].source == "ckpt-2"
        assert FakeRuntime.made[-1].files() == {"a.txt": b"a.txt"}
    model.requests.clear()
    resumed = await make(model).run(resume=loaded, at=0)
    assert roles(model.requests[0]["messages"]) == ["user"]
    assert FakeRuntime.made[-1].files() == {}
    with pytest.raises(ValueError, match="no checkpoint 'ckpt-9'"):
        await make(model).run(resume=loaded, at="ckpt-9")
    with pytest.raises(ValueError, match="no index 7"):
        await make(model).run(resume=loaded, at=7)


async def test_run_continues_a_trace_with_a_note(model):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        await interaction.checkpoint()
    model.requests.clear()
    trace = await agent.run(resume=reload(interaction.trace), note="and then?")
    assert trace.ok, trace.errors
    assert model.requests[0]["messages"][-1] == {"role": "user", "content": "and then?"}
    assert trace.task.data.prompt[-1].content == "and then?"
    assert trace.assistant_messages[-1].content == "done: and then?"


async def test_run_rejects_a_trace_ending_on_a_reply_without_note(model):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        await interaction.checkpoint()
    made = len(FakeRuntime.made)
    with pytest.raises(ValueError, match="note="):
        await agent.run(resume=interaction.trace)
    assert len(FakeRuntime.made) == made


async def test_a_borrowed_box_is_not_checkpointed(model):
    agent = make(model)
    async with (
        agent.provision(task()) as box,
        agent.interaction(task(), runtime=box) as interaction,
    ):
        await interaction.turn()
        with pytest.raises(ValueError, match="borrowed box"):
            await interaction.checkpoint()
    assert interaction.trace.checkpoints == []


def checkpointed(*checkpoints: vf.Checkpoint) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask.of(task()),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        checkpoints=list(checkpoints),
    )


async def test_resume_impossible_raises_before_provisioning(model, monkeypatch):
    trace = checkpointed(vf.Checkpoint(id="ckpt-9", turns=0))
    monkeypatch.setattr(
        agent_module, "restore_config", vf.runtimes.restore_config
    )  # the real one: subprocess runtimes cannot restore
    with pytest.raises(ValueError, match="cannot restore checkpoints"):
        await make(model).run(resume=trace)
    with pytest.raises(ValueError, match="no checkpoints"):
        await make(model, save="per_call").run(task())
    with pytest.raises(ValueError, match="SUPPORTS_RESUME"):
        async with make(model, harness=NoResumeHarness).interaction(resume=trace):
            pass
    # No checkpoint: the box cannot be restored, so only an explicit task resumes.
    with pytest.raises(ValueError, match="pass task="):
        await make(model).run(resume=checkpointed())
    with pytest.raises(ValueError, match="no checkpoints; drop at="):
        await make(model).run(resume=checkpointed(), task=task(), at=0)
    with pytest.raises(ValueError, match="not both"):
        await make(model).run(resume=trace, checkpoint="ckpt-9")
    assert FakeRuntime.made == []
    assert model.requests == []


async def test_a_trace_without_checkpoint_resumes_on_a_fresh_box_with_task(model):
    first = await make(model).run(task())
    assert first.checkpoints == []
    model.requests.clear()
    trace = await make(model).run(resume=reload(first), task=task(), note="more")
    assert trace.ok, trace.errors
    assert FakeRuntime.made[-1].source is None
    assert model.requests[0]["messages"][-1]["content"] == "more"


# -- interaction retries --------------------------------------------------------


async def test_interaction_retry_continues_from_the_latest_checkpoint(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    saves: list[tuple[vf.Trace, vf.Checkpoint]] = []
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.interaction(
        task(), on_save=lambda t, c: saves.append((t, c))
    ) as interaction:
        segment = await interaction.turn()
        assert segment.last_reply == f"done: {RETRY_NOTE}"
        failed = interaction.failed_attempts
    assert len(failed) == 1
    assert not failed[0].ok
    assert failed[0].is_completed
    assert failed[0].errors[-1].type == "SandboxError"
    final = interaction.trace
    assert final is not failed[0]
    assert final.ok, final.errors
    assert [error.type for error in final.errors] == ["SandboxError"]
    # The replacement box came from the checkpoint at call 2 (a.txt written, b.txt
    # not).
    assert FakeRuntime.made[-1].source == "ckpt-2"
    assert FakeRuntime.made[-1].files() == {"a.txt": b"a.txt"}
    assert EVENTS == ["setup", "restore"]
    retried = model.requests[2]["messages"]
    assert roles(retried) == ["user", "assistant", "tool", "user"]
    assert retried[-1]["content"] == RETRY_NOTE
    assert [(t is final, c.id) for t, c in saves] == [
        (False, "ckpt-1"),
        (False, "ckpt-2"),
        (True, "ckpt-3"),
        (True, "ckpt-4"),
    ]
    assert interaction.latest_checkpoint is saves[-1][1]
    assert [(c.id, c.turns, c.restored) for c in final.checkpoints] == [
        ("ckpt-2", 0, True),
        ("ckpt-3", 0, False),
        ("ckpt-4", 2, False),
    ]


async def test_a_continuation_links_its_source_and_carries_its_info(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.interaction(
        task(), on_trace=lambda t: t.info.setdefault("minted", t.id)
    ) as interaction:
        interaction.trace.info["note"] = {"seat": "lead"}
        await interaction.turn()
    (failed,) = interaction.failed_attempts
    retried = interaction.trace
    # The retry continues the failed attempt: linked, its info carried over before
    # any observer saw the replacement.
    assert failed.resumed_from is None
    assert retried.resumed_from == failed.id
    assert retried.info == {"minted": failed.id, "note": {"seat": "lead"}}
    assert retried.info["note"] is not failed.info["note"]
    resumed = await make(model).run(resume=reload(retried), note="more")
    assert resumed.resumed_from == retried.id
    assert resumed.info == retried.info
    assert "resumed_from" in resumed.to_record()
    assert "resumed_from" not in failed.to_record()


async def test_interaction_retry_without_checkpoints_reopens_the_opening(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, interaction_retries=2)
    async with agent.interaction(task()) as interaction:
        segment = await interaction.turn()
        assert segment.last_reply == "done: start"
    assert len(interaction.failed_attempts) == 1
    assert interaction.trace.ok, interaction.trace.errors
    assert EVENTS == ["setup", "setup"]
    # The first segment, nothing checkpointed: the same opening, the same prompt.
    assert roles(model.requests[2]["messages"]) == ["user"]


class DiesOnSecondLaunch(ScriptHarness):
    launches = 0

    async def launch(self, *args, **kwargs) -> ProgramResult:
        DiesOnSecondLaunch.launches += 1
        if DiesOnSecondLaunch.launches == 2:
            raise box_lost()
        return await super().launch(*args, **kwargs)


async def test_a_later_turn_retry_adds_its_input_after_an_earlier_checkpoint(model):
    DiesOnSecondLaunch.launches = 0
    agent = make(model, harness=DiesOnSecondLaunch, interaction_retries=1)
    async with agent.interaction(task(prompt=None)) as interaction:
        await interaction.turn("first")
        await interaction.checkpoint()
        segment = await interaction.turn("second")
        assert len(interaction.failed_attempts) == 1
    assert segment.last_reply == "done: second"
    assert interaction.trace.ok, interaction.trace.errors
    assert FakeRuntime.made[-1].source == "ckpt-1"
    retried = model.requests[-1]["messages"]
    # The checkpoint predates the failed turn: its message follows the retry note.
    assert [m["content"] for m in retried if m["role"] == "user"] == [
        "first",
        RETRY_NOTE,
        "second",
    ]


async def test_a_later_turn_with_no_checkpoint_is_not_retried(model):
    DiesOnSecondLaunch.launches = 0
    agent = make(model, harness=DiesOnSecondLaunch, interaction_retries=2)
    async with agent.interaction(task(prompt=None)) as interaction:
        first = await interaction.turn("first")
        assert first.last_reply == "done: first"
        await interaction.turn("second")
        # The opening box lacks the first turn's files: no sound state to resume.
        assert interaction.failed_attempts == []
    assert not interaction.trace.ok
    assert interaction.trace.errors[-1].type == "SandboxError"
    assert len(FakeRuntime.made) == 1


async def test_an_interaction_retry_keeps_pending_steers(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, interaction_retries=1)
    async with agent.interaction(task()) as interaction:
        receipt = await vf.steer(
            "mind the logs", trace_id=interaction.trace.id, sender="op"
        )
        assert receipt.queued, receipt.reason
        await interaction.turn()
        assert len(interaction.failed_attempts) == 1
    retried = [
        request
        for request in model.requests[2:]
        if any(m.get("content") == "mind the logs" for m in request["messages"])
    ]
    assert retried, "the replacement rollout must still deliver the steer"
    assert [steer.text for steer in interaction.trace.steers] == ["mind the logs"]


async def test_a_failed_model_call_is_retried_on_the_same_box(model):
    model.refuse.add(1)
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.interaction(task()) as interaction:
        segment = await interaction.turn()
        assert interaction.failed_attempts == []
    # The box survived: no replacement was provisioned or restored.
    assert len(FakeRuntime.made) == 1
    assert EVENTS == ["setup"]
    trace = interaction.trace
    assert trace.ok, trace.errors
    assert [error.type for error in trace.errors] == ["ProviderError"]
    assert segment.last_reply.startswith("done")


async def test_task_error_is_not_retried(model):
    FakeRuntime.doomed["b.txt"] = lambda: vf.TaskError("task broke")
    agent = make(model, save="per_call", interaction_retries=3)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        assert interaction.failed_attempts == []
    assert not interaction.trace.ok
    assert interaction.trace.errors[-1].type == "TaskError"
    assert len(FakeRuntime.made) == 1
    assert len(model.requests) == 2


def test_retry_classification():
    assert retryable_error(box_lost())
    io = vf.SandboxError("upload failed")
    io.kind = "io"
    assert retryable_error(io)
    account = vf.SandboxError("Payment required")
    account.kind = "account"
    assert not retryable_error(account)
    assert not retryable_error(vf.SandboxError("untyped"))
    assert retryable_error(vf.ProviderError("503"))
    assert retryable_error(vf.HarnessError("exited 1"))
    assert retryable_error(vf.InterceptionError("connection reset"))
    assert not retryable_error(vf.TaskError("setup failed"))
    assert not retryable_error(ValueError("bug"))


# -- the trace's task -----------------------------------------------------------


class Graded(vf.TaskConfig):
    threshold: float = 0.5


class GradedTask(vf.Task[vf.TaskData, vf.State, Graded]):
    pass


def test_trace_task_records_class_and_config():
    original = GradedTask(vf.TaskData(idx=3, prompt="hi"), Graded(threshold=0.9))
    record = vf.TraceTask.of(original)
    assert record.cls == f"{__name__}:GradedTask"
    assert record.config == Graded(threshold=0.9).model_dump(mode="json")
    wire = vf.TraceTask[vf.WireTaskData].model_validate_json(record.model_dump_json())
    rebuilt = wire.rebuild()
    assert type(rebuilt) is GradedTask
    assert rebuilt.config == Graded(threshold=0.9)
    assert rebuilt.data.model_dump() == original.data.model_dump()
    assert rebuilt.key == original.key


def test_a_trace_task_recorded_before_cls_still_loads():
    old = {"type": "GradedTask", "data": {"idx": 0, "prompt": "hi"}}
    loaded = vf.TraceTask[vf.WireTaskData].model_validate(old)
    assert loaded.cls is None and loaded.config is None
    assert "cls" not in loaded.model_dump(mode="json")
    with pytest.raises(ValueError, match="pass the task explicitly"):
        loaded.rebuild()


def test_a_local_task_class_is_not_rebuilt():
    class Local(vf.Task):
        pass

    record = vf.TraceTask.of(Local(vf.TaskData(idx=0, prompt="hi")))
    with pytest.raises(ValueError, match="cannot be imported"):
        record.rebuild()
