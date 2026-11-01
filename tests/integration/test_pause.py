"""A paused box carries on where it stopped. A scripted model drives the bash harness
through a tool call that sleeps and then a model call, and the box is paused inside one
of them (`PrimeBackend.pause`), held, and resumed (`resume`). The rollout finishes as if
never paused: the harness program waits through the pause, the sleeping tool completes,
and the model call the pause cut off is retried into the interception server's replay of
it, so the model answers each turn once and the trace keeps one branch."""

import asyncio
import json
import time

import pytest
from aiohttp import web
from grid import grid

from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.runtimes import PrimeBackend
from verifiers.v1.runtimes.base import _LIVE

HOLD = 10
"""Seconds the box stays paused."""

STEPS = [
    {"bash": "sleep 20 && echo slept"},
    {"bash": "echo 'hello world' > answer.txt && cat answer.txt"},
    {"text": "Done."},
]
"""The model's script, one reply per turn: the tool pause lands in the first turn's
sleep, the model pause while the second turn is being answered."""


class PausingModel:
    def __init__(self, where: str) -> None:
        self.where = where
        self.calls = [0] * len(STEPS)
        self.cycles: list[asyncio.Task] = []
        self.resumed_at: float | None = None
        self.arrived_at: list[float] = []

    async def cycle(self, delay: float = 0) -> None:
        await asyncio.sleep(delay)
        # The rollout's box: the one live Prime runtime in this process.
        (box,) = [b for b in _LIVE if isinstance(b, PrimeBackend) and not b.stopped]
        await box.pause()
        await asyncio.sleep(HOLD)
        await box.resume()
        self.resumed_at = time.monotonic()

    async def reply(self, body: dict) -> dict:
        turn = sum(message["role"] == "assistant" for message in body["messages"])
        self.calls[turn] += 1
        self.arrived_at.append(time.monotonic())
        step = STEPS[turn]
        if self.where == "tool" and turn == 0:
            self.cycles.append(asyncio.create_task(self.cycle(delay=3)))
        if self.where == "model" and turn == 1 and self.calls[turn] == 1:
            await self.cycle()
        if "text" in step:
            message = {"role": "assistant", "content": step["text"]}
            finish = "stop"
        else:
            call = {
                "id": f"call_{turn}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": step["bash"]}),
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


@pytest.fixture
async def pausing_model(request):
    """A scripted chat-completions upstream that pauses the rollout's box at
    `request.param` ("tool" or "model"). Yields the model and its client config."""
    model = PausingModel(request.param)
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
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield model, {"base_url": f"http://127.0.0.1:{port}/v1"}
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("harness,harness_runtime", grid(["bash"]), indirect=True)
@pytest.mark.parametrize("pausing_model", ["tool", "model"], indirect=True)
async def test_pause(run, harness, harness_runtime, pausing_model):
    model, client = pausing_model
    (episode,) = await run(
        "echo-agentic",
        harness=harness,
        runtime=harness_runtime,
        client=client,
        max_turns=len(STEPS),
        rollout_timeout=600,
    )
    await asyncio.gather(*model.cycles)
    assert model.resumed_at is not None, "the box was never paused"
    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    assert model.calls == [1] * len(STEPS)
    assert trace.num_branches == 1
    if model.where == "tool":
        # The sleep spanned the pause: its result came back only after the resume.
        assert model.arrived_at[1] > model.resumed_at
        assert "slept" in str(trace.tool_messages[0].content)
