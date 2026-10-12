"""Agent sessions: `Session`, `Agent.snapshot`, `session=` continuations, per-call
saves (`AgentConfig.save`) and interaction retries (`AgentConfig.interaction_retries`).

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
from verifiers.v1.agent_session import RETRY_NOTE, retryable_error
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult, SubprocessConfig
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

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

    async def checkpoint(self, trace: vf.Trace | None = None) -> str:
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
        if trace is not None:
            trace.record_checkpoint(checkpoint)
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


async def test_per_call_saves_reach_on_save(model):
    saves: list[vf.Session] = []
    agent = make(model, save="per_call")
    trace = await agent.run(task(), on_save=saves.append)
    assert trace.ok, trace.errors
    assert [save.calls for save in saves] == [1, 2, 3]
    assert [save.checkpoint for save in saves] == ["ckpt-1", "ckpt-2", "ckpt-3"]
    assert [checkpoint.id for checkpoint in trace.checkpoints] == [
        "ckpt-1",
        "ckpt-2",
        "ckpt-3",
    ]
    # Each save is taken before the harness runs the response's tool calls: the box
    # holds the files of earlier calls only, and the conversation leaves the pending
    # tool call out.
    assert FakeRuntime.store["ckpt-1"] == {}
    assert FakeRuntime.store["ckpt-2"] == {"a.txt": b"a.txt"}
    assert FakeRuntime.store["ckpt-3"] == {"a.txt": b"a.txt", "b.txt": b"b.txt"}
    assert [m.role for m in saves[0].messages] == ["user"]
    assert [m.role for m in saves[1].messages] == ["user", "assistant", "tool"]
    final = saves[2].messages
    assert [m.role for m in final][-1] == "assistant"
    assert final[-1].content == "done: start"
    assert saves[2].task_type.endswith(":NotesTask")
    assert "prompt" not in saves[2].task_data


async def test_a_refused_checkpoint_is_tried_again(model, monkeypatch):
    monkeypatch.setattr(rollout_module, "SAVE_RETRY_DELAY", 0.0)
    FakeRuntime.failing_checkpoints = {2}
    saves: list[vf.Session] = []
    agent = make(model, save="per_call")
    trace = await agent.run(task(), on_save=saves.append)
    assert trace.ok, trace.errors
    assert [save.calls for save in saves] == [1, 2, 3]


async def test_a_checkpoint_refused_every_time_is_skipped(model, monkeypatch):
    monkeypatch.setattr(rollout_module, "SAVE_RETRY_DELAY", 0.0)
    FakeRuntime.failing_checkpoints = {2, 3, 4}
    saves: list[vf.Session] = []
    agent = make(model, save="per_call")
    trace = await agent.run(task(), on_save=saves.append)
    assert trace.ok, trace.errors
    assert [save.calls for save in saves] == [1, 3]
    assert [checkpoint.id for checkpoint in trace.checkpoints] == ["ckpt-1", "ckpt-2"]


async def test_on_save_failure_does_not_fail_the_run(model):
    def broken(save: vf.Session) -> None:
        raise OSError("disk full")

    trace = await make(model, save="per_call").run(task(), on_save=broken)
    assert trace.ok, trace.errors


# -- snapshot and reopen --------------------------------------------------------


async def test_snapshot_reopen_round_trip(model):
    agent = make(model)
    async with agent.interaction(task(prompt=None)) as interaction:
        segment = await interaction.turn("first")
        assert segment.last_reply == "done: first"
        snapshot = await agent.snapshot(interaction)
    assert snapshot.checkpoint == "ckpt-1"
    assert snapshot.calls == 3
    assert [m.role for m in snapshot.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    session = vf.Session.model_validate_json(snapshot.model_dump_json())
    assert session == snapshot

    EVENTS.clear()
    model.requests.clear()
    # The conversation ends on the agent's reply: the first turn continues it.
    async with agent.interaction(session=session) as reopened:
        segment = await reopened.turn("second")
        assert segment.last_reply == "done: second"
        assert reopened.calls == 4
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


async def test_run_continues_a_session_with_a_note(model):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        session = await agent.snapshot(interaction)
    model.requests.clear()
    trace = await agent.run(session=session, note="and then?")
    assert trace.ok, trace.errors
    assert model.requests[0]["messages"][-1] == {"role": "user", "content": "and then?"}
    assert trace.task.data.prompt[-1].content == "and then?"
    assert trace.assistant_messages[-1].content == "done: and then?"


async def test_run_rejects_a_session_ending_on_a_reply_without_note(model):
    agent = make(model)
    async with agent.interaction(task()) as interaction:
        await interaction.turn()
        session = await agent.snapshot(interaction)
    made = len(FakeRuntime.made)
    with pytest.raises(ValueError, match="note="):
        await agent.run(session=session)
    assert len(FakeRuntime.made) == made


async def test_restore_impossible_raises_before_provisioning(model, monkeypatch):
    session = vf.Session.of(
        task(), checkpoint="ckpt-9", messages=[vf.UserMessage(content="hi")], calls=0
    )
    monkeypatch.setattr(
        agent_module, "restore_config", vf.runtimes.restore_config
    )  # the real one: subprocess runtimes cannot restore
    with pytest.raises(ValueError, match="cannot restore checkpoints"):
        await make(model).run(session=session)
    with pytest.raises(ValueError, match="no checkpoints"):
        await make(model, save="per_call").run(task())
    with pytest.raises(ValueError, match="SUPPORTS_RESUME"):
        async with make(model, harness=NoResumeHarness).interaction(session=session):
            pass
    assert FakeRuntime.made == []
    assert model.requests == []


# -- interaction retries --------------------------------------------------------


async def test_interaction_retry_continues_from_the_latest_save(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    saves: list[vf.Session] = []
    agent = make(model, save="per_call", interaction_retries=1)
    async with agent.interaction(task(), on_save=saves.append) as interaction:
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
    # The replacement box came from the save at call 2 (a.txt written, b.txt not).
    assert FakeRuntime.made[-1].source == "ckpt-2"
    assert FakeRuntime.made[-1].files() == {"a.txt": b"a.txt"}
    assert EVENTS == ["setup", "restore"]
    retried = model.requests[2]["messages"]
    assert roles(retried) == ["user", "assistant", "tool", "user"]
    assert retried[-1]["content"] == RETRY_NOTE
    assert [save.calls for save in saves] == [1, 2, 3, 4]
    assert interaction.latest_save is saves[-1]
    assert interaction.latest_save.calls == 4


async def test_interaction_retry_without_saves_reopens_the_opening(model):
    FakeRuntime.doomed["b.txt"] = box_lost
    agent = make(model, interaction_retries=2)
    async with agent.interaction(task()) as interaction:
        segment = await interaction.turn()
        assert segment.last_reply == "done: start"
    assert len(interaction.failed_attempts) == 1
    assert interaction.trace.ok, interaction.trace.errors
    assert EVENTS == ["setup", "setup"]
    # The first segment, nothing saved: the same opening, the same prompt.
    assert roles(model.requests[2]["messages"]) == ["user"]


class DiesOnSecondLaunch(ScriptHarness):
    launches = 0

    async def launch(self, *args, **kwargs) -> ProgramResult:
        DiesOnSecondLaunch.launches += 1
        if DiesOnSecondLaunch.launches == 2:
            raise box_lost()
        return await super().launch(*args, **kwargs)


async def test_a_later_turn_with_no_save_is_not_retried(model):
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


def test_session_is_versioned():
    data = vf.Session.of(task(), checkpoint=None, messages=[], calls=0).model_dump(
        mode="json"
    )
    assert data["v"] == 1
    with pytest.raises(ValueError):
        vf.Session.model_validate({**data, "v": 2})
    rebuilt = vf.Session.model_validate(data).rebuild_task()
    assert type(rebuilt) is NotesTask
    assert rebuilt.data.prompt is None
