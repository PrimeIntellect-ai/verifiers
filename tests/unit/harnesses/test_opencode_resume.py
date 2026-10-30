"""A cut-off OpenCode run continues its own trace from its latest checkpoint.

OpenCode keeps its session in its data directory on the box, which a checkpoint keeps:
a durable call cut mid-run (flow's process killed) continues on a box restored from its
latest per-call checkpoint by resuming that session (`ACPConfig.session_file`), the same
trace growing in place, instead of starting the draw over (2026-10-30: relaunched GLM
evals ran every in-flight OpenCode draw from scratch).

The real-binary test needs `VF_OPENCODE_BIN` (an `opencode` 2.1.x executable) and `uv`;
it drives the real ACP runner and OpenCode against a scripted local chat server on the
subprocess runtime, whose fake checkpoints keep the box's work directory and OpenCode's
state."""

import asyncio
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from aiohttp import web

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
import verifiers.v1.harnesses.opencode.harness as opencode_module
import verifiers.v1.rollout as rollout_module
from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.runtimes import ProgramResult, SubprocessConfig
from verifiers.v1.runtimes.backends.subprocess import SubprocessBackend
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage
from verifiers.v1.utils.retries import RETRY_NOTE

# -- which session a continued run attaches to ---------------------------------------


class Box:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    async def run(self, argv, env) -> ProgramResult:
        path = argv[-1].split()[1].strip("'")
        if path in self.files:
            return ProgramResult(exit_code=0, stdout=self.files[path], stderr="")
        return ProgramResult(exit_code=1, stdout="", stderr="")


def config(prompt) -> ACPConfig:
    return ACPConfig(env={}, command=[], prompt=prompt, session_file="/s/acp-session")


async def continued(prompt, files: dict[str, str]) -> ACPConfig:
    harness = SimpleNamespace(config=SimpleNamespace(id="opencode"))
    acp = config(prompt)
    await ACPHarness._continue_session(harness, acp, Box(files))
    return acp


async def test_a_restored_session_is_sent_only_the_messages_past_its_turns():
    note = UserMessage(content=RETRY_NOTE)
    conversation = [
        UserMessage(content="task"),
        AssistantMessage(content="", tool_calls=[]),
        ToolMessage(tool_call_id="c", content="ok"),
        note,
    ]
    acp = await continued(conversation, {"/s/acp-session": "ses_1\n"})
    assert acp.continues and acp.load_session_id == "ses_1"
    assert acp.prompt == [note]
    # Cut before the first reply: the session holds the task's prompt.
    acp = await continued([UserMessage(content="task"), note], {"/s/acp-session": "s"})
    assert acp.prompt == [note]
    # Ending on a tool result: the host's note says what happened.
    acp = await continued(conversation[:3], {"/s/acp-session": "s"})
    assert acp.prompt == [UserMessage(content=RETRY_NOTE)]


async def test_a_box_without_a_recorded_session():
    """Only user messages: a new session opens on them. Turns: the box is from a run
    started before its session was recorded, and its agent's latest session continues."""
    acp = await continued("task", {})
    assert not acp.continues and acp.prompt == "task"
    users = [UserMessage(content="task"), UserMessage(content=RETRY_NOTE)]
    acp = await continued(users, {})
    assert not acp.continues and acp.prompt == users
    acp = await continued(
        [UserMessage(content="task"), AssistantMessage(content="x"), users[1]], {}
    )
    assert acp.continues and acp.load_session_id is None
    assert acp.prompt == [users[1]]


# -- a real OpenCode run, cut and continued ------------------------------------------

STATE: Path
BOX: Path


class Boxes(SubprocessBackend):
    """A subprocess box whose checkpoints keep its work directory and OpenCode's state
    root, as a Prime checkpoint keeps the whole disk. Every box has the same work
    directory, as a restored Prime box keeps the image's."""

    store: ClassVar[dict[str, dict[str, bytes]]] = {}
    made: ClassVar[list["Boxes"]] = []

    def __init__(self, config, name=None) -> None:
        self.source = getattr(config, "checkpoint", None)
        super().__init__(
            SubprocessConfig(**config.model_dump(exclude={"checkpoint"})), name
        )
        Boxes.made.append(self)

    async def start(self) -> None:
        for root in (BOX, STATE):
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True)
        self.workdir = BOX
        self.info.id = str(BOX)
        for path, data in self.store.get(self.source, {}).items():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data)

    @property
    def restored(self) -> bool:
        return self.source is not None

    async def checkpoint(self) -> str:
        checkpoint = f"ckpt-{len(self.store) + 1}"
        self.store[checkpoint] = {
            str(path): path.read_bytes()
            for root in (BOX, STATE)
            for path in root.rglob("*")
            if path.is_file()
        }
        return checkpoint

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.store.pop(checkpoint_id, None)


class RestoreConfig(SubprocessConfig):
    checkpoint: str | None = None


class Script:
    """The model: write a.txt, then b.txt, then stop. `cut` blocks the request after
    the second write (flow's process is killed while the model thinks)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.cut = asyncio.Event()
        self.cutting = True

    def step(self, messages: list[dict]) -> tuple[str | None, dict | None]:
        written = sum(m["role"] == "tool" for m in messages)
        if written < 2:
            name = "ab"[written] + ".txt"
            path = str(BOX / name)
            return f"Writing {name}.", {"path": path, "content": name}
        return "Done.", None

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        text, args = self.step(body["messages"])
        if args is None and self.cutting:
            self.cut.set()
            await asyncio.Event().wait()  # until the run is cancelled
        n = len(self.requests)
        message: dict = {"role": "assistant", "content": text}
        if args is not None:
            message["tool_calls"] = [
                {
                    "id": f"call_{n}",
                    "type": "function",
                    "function": {"name": "write", "arguments": json.dumps(args)},
                }
            ]
        finish = "tool_calls" if args is not None else "stop"
        usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        if not body.get("stream"):
            return web.json_response(
                {
                    "id": f"c{n}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": usage,
                }
            )
        base = {"id": f"c{n}", "object": "chat.completion.chunk", "created": 0}
        delta: dict = {"role": "assistant", "content": text}
        if args is not None:
            delta["tool_calls"] = [{"index": 0, **message["tool_calls"][0]}]
        events = [
            {**base, "choices": [{"index": 0, "delta": delta}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {**base, "choices": [], "usage": usage},
        ]
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        for event in events:
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response


@pytest.fixture
async def script():
    model = Script()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", model.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    model.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    yield model
    await runner.cleanup()


@pytest.fixture
def opencode(tmp_path, monkeypatch):
    """OpenCode from `VF_OPENCODE_BIN` where the harness runs it; its state root under
    the test's directory, kept by the boxes' checkpoints."""
    global STATE, BOX
    binary = os.environ.get("VF_OPENCODE_BIN")
    if not binary or not shutil.which("uv"):
        pytest.skip("needs VF_OPENCODE_BIN (opencode 2.1.x) and uv")
    STATE, BOX = tmp_path / "state", tmp_path / "box"
    version = "2.1.4"
    bin_dir = tmp_path / "opencode" / version / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "opencode").symlink_to(binary)
    monkeypatch.setattr(opencode_module, "OPENCODE_DIR", str(tmp_path / "opencode"))
    monkeypatch.setattr(opencode_module, "STATE_ROOT", str(STATE))
    # OpenCode is the binary above: only the ACP runner is set up.
    monkeypatch.setattr(opencode_module.OpenCodeHarness, "setup", ACPHarness.setup)
    # The fake boxes stand in for containers.
    monkeypatch.setattr(opencode_module.OpenCodeHarness, "NEEDS_CONTAINER", False)
    Boxes.store, Boxes.made = {}, []
    monkeypatch.setattr(
        rollout_module, "make_backend", lambda config, name=None: Boxes(config, name)
    )
    monkeypatch.setattr(
        agent_module,
        "restore_config",
        lambda config, checkpoint: RestoreConfig(
            **config.model_dump(), checkpoint=checkpoint
        ),
    )
    monkeypatch.setenv("SCRIPTED_KEY", "scripted")
    return version


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "unrecorded"])
async def test_a_cut_off_opencode_run_continues_its_trace_from_its_checkpoint(
    script, opencode, tmp_path, recorded
):
    """`unrecorded`: the box is from a run started before the runner recorded its
    session, as an in-flight draw of an earlier release: OpenCode's latest continues."""
    agent = vf.Agent(
        vf.AgentConfig(
            model="scripted",
            client={
                "type": "eval",
                "base_url": script.url,
                "api_key_var": "SCRIPTED_KEY",
            },
            runtime={"type": "subprocess"},
            harness={"id": "opencode", "version": opencode},
            save="per_call",
            save_interval=0,
        )
    )
    task = vf.Task(vf.TaskData(idx=0, prompt="Write a.txt, then b.txt."))
    store = vf.TraceDir(tmp_path / "traces")
    with vf.use_durable(vf.Durable(store)):
        run = asyncio.create_task(agent.run(task, key="solve"))
        cut_seen = asyncio.create_task(script.cut.wait())
        await asyncio.wait({run, cut_seen}, timeout=300, return_when="FIRST_COMPLETED")
        if run.done():
            trace = run.result()
            pytest.fail(f"the run ended before its cut: {trace.errors}")
        assert cut_seen.done(), "the model never got its third request"
        run.cancel()  # flow's process killed mid-turn
        with pytest.raises(asyncio.CancelledError):
            await run
        [cut] = [
            vf.WireTrace.model_validate_json(path.read_bytes())
            for path in (tmp_path / "traces").glob("*.json")
        ]
        assert not cut.is_completed
        assert [c.id for c in cut.checkpoints] == ["ckpt-1", "ckpt-2"]
        cut_turns = cut.num_turns
        if not recorded:
            for files in Boxes.store.values():
                for path in [p for p in files if p.endswith("/acp-session")]:
                    del files[path]
        before = len(script.requests)
        script.cutting = False

        resumed = await asyncio.wait_for(agent.run(task, key="solve"), 300)

    assert resumed.ok, resumed.errors
    assert resumed.id == cut.id
    assert Boxes.made[-1].source == "ckpt-2"
    [entry] = resumed.info["resumes"]
    assert entry["checkpoint"] == "ckpt-2"
    # The continued session replays its own turns: the first request after the cut
    # carries the task, the first write and its result, then the host's note.
    first = script.requests[before]["messages"]
    users = [m["content"] for m in first if m["role"] == "user"]
    assert any("Write a.txt" in json.dumps(u) for u in users)
    assert sum(m["role"] == "tool" for m in first) == 1
    assert "infrastructure failure" in json.dumps(first[-1])
    # It went on from there: b.txt written once, then the final reply.
    assert len(script.requests) - before == 2
    # The final reply's checkpoint holds the box as the run left it.
    final = Boxes.store[resumed.checkpoints[-1].id]
    assert final[str(BOX / "a.txt")] == b"a.txt"
    assert final[str(BOX / "b.txt")] == b"b.txt"
    assert resumed.num_turns > cut_turns >= 1
    assert resumed.root_reply == "Done."
