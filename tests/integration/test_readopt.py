"""A rollout outlives the process that ran it: a new process takes it over in its live
box. A client runs a rollout through an env server process, holding the stream
(`EpisodeAssembly`) the way an orchestrator does; it keeps a copy of the trace and its
restore point as turn 2 is asked (an orchestrator checkpoint). The env server is then
SIGKILLed — while a tool sleeps, or while a model call is in flight — and another env
server continues the rollout from that copy (`RunRequest.restore`): it adopts the box,
the tunnel and the secrets its harness program holds, and waits on that program, which
never noticed. The program's background process keeps its PID, and the turns it took
after the copy (the gap) are recorded as context, not as samples. With the box gone, the
restore starts over."""

import asyncio
import contextlib
import copy
import functools
import json
import multiprocessing as mp
import socket
import time
from pathlib import Path

import pytest
from aiohttp import web
from conftest import _eval_config
from grid import grid, runtime_config
from prime_sandboxes import AsyncSandboxClient

from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.runtimes.backends.prime import delete_boxes
from verifiers.v1.serve import EnvClient, EnvServer, EpisodeAssembly, RestoreRequest
from verifiers.v1.serve.pool import env_config_data, serve_env
from verifiers.v1.types import content_text
from verifiers.v1.utils.loaders import load_environment
from verifiers.v1.utils.logging import setup_logging

START = "setsid sleep 100000 >/dev/null 2>&1 < /dev/null & echo $! > bg.pid; cat bg.pid"
"""Starts the background process whose PID must survive."""
CHECK = (
    'if kill -0 "$(cat bg.pid)" 2>/dev/null && grep -q 100000 /proc/$(cat bg.pid)/cmdline; '
    'then echo "alive $(cat bg.pid)"; else echo gone; fi; '
    "echo 'hello world' > answer.txt"
)
SCRIPTS = {
    "tool": [START, "echo one", "echo two", "sleep 30 && echo slept", CHECK, None],
    "model": [START, "echo one", "echo two", CHECK, None],
}
"""The model's replies by turn (None ends the run). The copy is taken as turn 2 is
asked, so turns 2 and up are the gap; the process dies in turn 3."""
COPY_TURN = 2
KILL_TURN = 3


class ScriptedModel:
    """Answers each rollout by its turn (the replies so far); `rollouts` run the same
    script at once. The first to ask turn `COPY_TURN` waits for the copy (with `copy`);
    once every rollout asked turn `KILL_TURN`, the process is due to die."""

    def __init__(self, where: str, *, copy: bool = True, rollouts: int = 1) -> None:
        self.where = where
        self.script = SCRIPTS[where]
        self.copy = copy
        self.rollouts = rollouts
        self.calls = [0] * len(self.script)
        self.copy_asked = asyncio.Event()
        self.copied = asyncio.Event()
        self.kill = asyncio.Event()
        self.release = asyncio.Event()

    async def reply(self, body: dict) -> dict:
        turn = sum(message["role"] == "assistant" for message in body["messages"])
        self.calls[turn] += 1
        if self.copy and turn == COPY_TURN and self.calls[turn] == 1:
            self.copy_asked.set()
            await self.copied.wait()
        if turn == KILL_TURN and self.calls[turn] <= self.rollouts:
            if self.calls[turn] == self.rollouts:
                self.kill.set()
            if self.where == "model":
                # In flight when the process dies: it never gets an answer.
                await self.release.wait()
                raise web.HTTPServiceUnavailable()
        command = self.script[turn]
        if command is None:
            message = {"role": "assistant", "content": "Done."}
            finish = "stop"
        else:
            call = {
                "id": f"call_{turn}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
            finish = "tool_calls"
        return {
            "id": f"chatcmpl-{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "scripted"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@contextlib.asynccontextmanager
async def serve_model(model: ScriptedModel):
    dialect = ChatDialect()

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        raw = await model.reply(body)
        if not dialect.streaming(body):
            return web.json_response(raw)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in dialect.stream_events(raw):
            await response.write(event)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
        }
    finally:
        await runner.cleanup()


def free_address() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


async def copy_when_live(assembly: EpisodeAssembly) -> tuple[dict, dict]:
    """The trace and restore point as an orchestrator checkpoint would save them, once
    the point carries the live box."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        for trace_id, point in assembly.restores.items():
            if point.get("live"):
                trace = assembly.traces[trace_id]
                return (
                    copy.deepcopy({k: v for k, v in trace.items() if k != "pending"}),
                    copy.deepcopy(point),
                )
        await asyncio.sleep(0.5)
    raise AssertionError(f"no live restore point streamed: {assembly.restores}")


@pytest.mark.parametrize("harness,harness_runtime", grid(["bash"]), indirect=True)
@pytest.mark.parametrize(
    "where,gone",
    [("tool", False), ("model", False), ("tool", True)],
    ids=["tool", "model", "box-gone"],
)
async def test_readopt(harness, harness_runtime, where, gone, tmp_path: Path):
    model = ScriptedModel(where)
    box: str | None = None
    async with serve_model(model) as client_config:
        runtime = {**runtime_config("prime", "bash"), "labels": ["claude-pause-test"]}
        config = _eval_config(
            "echo-agentic",
            harness=harness,
            runtime=runtime,
            client=client_config,
            max_turns=len(model.script),
            rollout_timeout=600,
        )
        kwargs = {
            "client": config.client,
            "model": config.model,
            "sampling": config.sampling,
        }
        task_data = next(iter(load_environment(config.env).taskset)).data
        task_data = task_data.model_dump(mode="json")
        address = free_address()
        first = mp.get_context("spawn").Process(
            target=serve_env,
            kwargs={
                "max_workers": 1,
                "address": address,
                "config_data": env_config_data(config.env),
                "log_setup": functools.partial(
                    setup_logging, "INFO", str(tmp_path / "first.log"), False
                ),
            },
        )
        first.start()
        client = EnvClient(address=address)
        serving: asyncio.Task | None = None
        try:
            await client.wait_for_server_startup(timeout=60)
            assembly = EpisodeAssembly()
            running = asyncio.create_task(
                client.run(task_data=task_data, assembly=assembly, **kwargs)
            )
            await model.copy_asked.wait()
            trace, point = await copy_when_live(assembly)
            box = point["live"]["sandbox_id"]
            model.copied.set()
            await model.kill.wait()
            if where == "tool":
                await asyncio.sleep(8)  # inside the 30 s sleep
            first.kill()
            first.join()
            running.cancel()
            with contextlib.suppress(BaseException):
                await running
            if gone:
                assert await delete_boxes([box]) == [box]

            second = EnvServer(config.env, address=free_address())
            serving = asyncio.create_task(second.run())
            restarted = EnvClient(address=second.address)
            try:
                await restarted.wait_for_server_startup(timeout=60)
                episode = await restarted.run(
                    task_data=task_data,
                    restore=RestoreRequest(trace=trace, point=point),
                    **kwargs,
                )
            finally:
                await restarted.close()
        finally:
            model.release.set()
            await client.close()
            if first.is_alive():
                first.kill()
            if serving is not None:
                serving.cancel()
                with contextlib.suppress(BaseException):
                    await serving
            if box is not None:
                await delete_boxes([box])

    (result,) = episode.traces
    assert result.ok, [error.message for error in result.errors]
    assert result.reward == 1.0
    assert result.num_branches == 1
    started = content_text(result.tool_messages[0].content).split()[0]
    checked = content_text(result.tool_messages[-1].content)
    assert checked.strip() == f"alive {started}"
    if gone:
        # Nothing left to continue from: a new run of the task.
        assert result.id != trace["id"]
        assert result.agent.runtime.id != box
        return
    assert result.id == trace["id"]
    assert result.agent.runtime.id == box
    # The gap's turns ran in the first process; the new one recorded them as context.
    replies = [n for n in result.nodes if n.message.role == "assistant"]
    gap = KILL_TURN - COPY_TURN + (where == "tool")
    assert [n.sampled for n in replies] == (
        [True] * COPY_TURN
        + [False] * gap
        + [True] * (len(model.script) - COPY_TURN - gap)
    )
    # Each turn went upstream once, but the one in flight when the process died.
    in_flight = [KILL_TURN] if where == "model" else []
    assert model.calls == [1 + (turn in in_flight) for turn in range(len(model.script))]


@pytest.mark.parametrize("harness,harness_runtime", grid(["bash"]), indirect=True)
async def test_suspend_on_shutdown(
    harness, harness_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An env server stopped mid-rollout (SIGTERM) with `VF_SUSPEND_DIR` set pauses the
    boxes it would delete and records them. Its client detached first, so cancelling
    its runs did not delete them either. Both rollouts shared the server's tunnel; a
    pool of two workers re-adopts them, both on the one worker that serves the tunnel,
    and they finish in their paused boxes with their background processes."""
    model = ScriptedModel("tool", copy=False, rollouts=2)
    suspended = tmp_path / "suspended"
    boxes: list[str] = []
    async with serve_model(model) as client_config:
        runtime = {**runtime_config("prime", "bash"), "labels": ["claude-pause-test"]}
        config = _eval_config(
            "echo-agentic",
            harness=harness,
            runtime=runtime,
            client=client_config,
            max_turns=len(model.script),
            rollout_timeout=600,
        )
        kwargs = {
            "client": config.client,
            "model": config.model,
            "sampling": config.sampling,
        }
        task_data = next(iter(load_environment(config.env).taskset)).data
        task_data = task_data.model_dump(mode="json")
        spawn = mp.get_context("spawn")
        first_address = free_address()
        monkeypatch.setenv("VF_SUSPEND_DIR", str(suspended))
        first = spawn.Process(
            target=serve_env,
            kwargs={
                "max_workers": 1,
                "address": first_address,
                "config_data": env_config_data(config.env),
                "log_setup": functools.partial(
                    setup_logging, "INFO", str(tmp_path / "first.log"), False
                ),
            },
        )
        first.start()
        monkeypatch.delenv("VF_SUSPEND_DIR")
        address = free_address()
        pool = spawn.Process(
            target=serve_env,
            kwargs={
                "max_workers": 2,
                "elastic": False,
                "address": address,
                "config_data": env_config_data(config.env),
                "log_setup": functools.partial(
                    setup_logging, "INFO", str(tmp_path / "pool.log"), False
                ),
            },
        )
        client = EnvClient(address=first_address)
        restarted = EnvClient(address=address)
        try:
            await client.wait_for_server_startup(timeout=60)
            assemblies = [EpisodeAssembly(), EpisodeAssembly()]
            running = [
                asyncio.create_task(
                    client.run(task_data=task_data, assembly=assembly, **kwargs)
                )
                for assembly in assemblies
            ]
            await model.kill.wait()
            await asyncio.sleep(3)  # both inside their 30 s sleep
            copies = []
            for assembly in assemblies:
                ((trace_id, point),) = assembly.restores.items()
                trace = assembly.traces[trace_id]
                copies.append(
                    (
                        copy.deepcopy(
                            {k: v for k, v in trace.items() if k != "pending"}
                        ),
                        copy.deepcopy(point),
                    )
                )
            boxes = [point["live"]["sandbox_id"] for _, point in copies]
            # A shutdown: the client detaches, cancels its runs, then the server stops.
            client.detach()
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            first.terminate()
            first.join(timeout=120)
            assert first.exitcode is not None, "the server did not stop"

            (manifest,) = suspended.glob("*.jsonl")
            lines = [json.loads(line) for line in manifest.read_text().splitlines()]
            assert sorted(line["sandbox_id"] for line in lines) == sorted(boxes)
            assert all(line["paused"] for line in lines)
            assert len({line["tunnel_id"] for line in lines}) == 1
            assert {line["trace_id"] for line in lines} == {t["id"] for t, _ in copies}
            async with AsyncSandboxClient() as sandboxes:
                assert [(await sandboxes.get(box)).status for box in boxes] == [
                    "PAUSED"
                ] * 2

            pool.start()
            await restarted.wait_for_server_startup(timeout=120)
            episodes = await asyncio.gather(
                *(
                    restarted.run(
                        task_data=task_data,
                        restore=RestoreRequest(trace=trace, point=point),
                        **kwargs,
                    )
                    for trace, point in copies
                )
            )
        finally:
            model.release.set()
            await client.close()
            await restarted.close()
            for process in (first, pool):
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
            await delete_boxes(boxes)

    for episode, (trace, point) in zip(episodes, copies, strict=True):
        (result,) = episode.traces
        assert result.id == trace["id"]
        assert result.ok, [error.message for error in result.errors]
        assert result.reward == 1.0
        assert result.agent.runtime.id == point["live"]["sandbox_id"]
        started = content_text(result.tool_messages[0].content).split()[0]
        assert content_text(result.tool_messages[-1].content).strip() == (
            f"alive {started}"
        )
    # One worker served the shared tunnel for both.
    log = (tmp_path / "pool.log").read_text()
    assert log.count(": adopted") == 1, log
    assert log.count("rollout adopt:") == 2
