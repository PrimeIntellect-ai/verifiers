"""A model call the provider fails transiently is retried in place by the interception
server (`ProviderRetryConfig`), before anything of the turn commits, until it succeeds or
the time budget runs out; any other failure is final at once. Each failure mode is one
the TB2 campaign met on 2026-10-14: Cloudflare 502s, 503 "reset before headers", 500s,
dropped connections, read timeouts, streams cut short, and, the one that cost rollouts,
a call that failed after its harness connection had dropped. Served through the real
interception server by a scripted endpoint that fails on cue; no model, no API key."""

import asyncio
import json
import time
from typing import Self

import httpx
import pytest
from aiohttp import web
from openai import AsyncOpenAI
from test_agent_resume import (  # noqa: F401 - fixtures
    TOOLS,
    NotesTask,
    ScriptHarness,
    fake_boxes,
    reply,
)

import verifiers.v1 as vf
from verifiers.v1.interception import server
from verifiers.v1.runtimes import ProgramResult

CLOUDFLARE_502 = {
    "type": "https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-502/",
    "title": "Error 502: Bad gateway",
    "status": 502,
    "error_name": "origin_bad_gateway",
}
RESET_BEFORE_HEADERS = (
    "upstream connect error or disconnect/reset before headers. "
    "reset reason: connection termination"
)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(server, "RETRY_BACKOFF_SECONDS", 0.01)


class Provider:
    """A scripted chat-completions endpoint. `script` says how each request in turn
    is answered (then the scripted reply): an HTTP status, `"reset"` (the connection
    closes before any header), `"truncate"` (a stream cut off before its end), or a
    `(delay, failure)` pair that waits first (a queue)."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.requests: list[dict] = []
        self.times: list[float] = []
        self.url = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        self.times.append(time.monotonic())
        failure = self.script.pop(0) if self.script else None
        if isinstance(failure, tuple):
            delay, failure = failure
            await asyncio.sleep(delay)
        if failure == "reset":
            assert request.transport is not None
            request.transport.abort()
            return web.Response()
        if isinstance(failure, int):
            text = (
                json.dumps(CLOUDFLARE_502)
                if failure == 502
                else RESET_BEFORE_HEADERS
                if failure == 503
                else json.dumps({"error": {"message": f"failure {failure}"}})
            )
            headers = {"Retry-After": "0.3"} if failure == 429 else None
            return web.Response(text=text, status=failure, headers=headers)
        completion = reply(body["messages"])
        if failure == "leak":
            # the provider's parser missed the model's tool call: no content, no
            # tool_calls, the raw markup left in the reasoning
            completion["choices"][0]["message"] = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I will list the files.<tool_call>bash"
                "<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>",
            }
            completion["choices"][0]["finish_reason"] = "stop"
        if failure == "dropped":
            # the provider generated the model's tool call but sent none of it: no
            # content, no tool_calls, its usage counting the output past the reasoning
            completion["choices"][0]["message"] = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let me list the files.",
            }
            completion["choices"][0]["finish_reason"] = "stop"
            completion["usage"] = {
                "prompt_tokens": 20,
                "completion_tokens": 85,
                "total_tokens": 105,
                "completion_tokens_details": {"reasoning_tokens": 6},
            }
        if failure == "eos":
            # the provider answered with one end-of-turn token: no content, no tool
            # calls, no reasoning
            completion["choices"][0]["message"] = {"role": "assistant", "content": ""}
            completion["choices"][0]["finish_reason"] = "stop"
            completion["usage"] = {
                "prompt_tokens": 20,
                "completion_tokens": 1,
                "total_tokens": 21,
            }
        if failure == "done":
            # a model that ends its run with an empty reply after reasoning: kept
            completion["choices"][0]["message"] = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "All done.",
            }
            completion["choices"][0]["finish_reason"] = "stop"
            completion["usage"] = {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "completion_tokens_details": {"reasoning_tokens": 4},
            }
        if not body.get("stream"):
            return web.json_response(completion)
        message = completion["choices"][0]["message"]
        delta = {"role": "assistant", "content": message.get("content")}
        if message.get("reasoning_content"):
            delta["reasoning_content"] = message["reasoning_content"]
        if calls := message.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **c} for i, c in enumerate(calls)]
        chunk = {
            "id": "scripted",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "scripted",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None
                    if failure in ("cut", "error-event")
                    else completion["choices"][0]["finish_reason"],
                }
            ],
        }
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        if failure == "keepalives":
            # the provider holds the turn: keepalive comments only, then it lets go
            for _ in range(8):
                await stream.write(b": keepalive\n\n")
                await asyncio.sleep(0.1)
            return stream
        if failure == "slow":
            # a long generation: each event comes well inside the read timeout, the
            # whole turn takes longer than it
            for _ in range(3):
                await stream.write(
                    b'data: {"id": "scripted", "object": "chat.completion.chunk", '
                    b'"created": 0, "model": "scripted", "choices": [{"index": 0, '
                    b'"delta": {"reasoning_content": "."}, "finish_reason": null}]}\n\n'
                )
                await asyncio.sleep(0.2)
        await stream.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if failure == "truncate":
            assert request.transport is not None
            request.transport.abort()
            return stream
        if failure == "cut":
            # the provider closes the stream cleanly, its terminal event never sent
            return stream
        if failure == "error-event":
            # the provider fails the turn inside the stream, then ends it as if done
            error = {"error": {"message": "Service error. Please retry shortly."}}
            await stream.write(f"data: {json.dumps(error)}\n\n".encode())
            await stream.write(b"data: [DONE]\n\n")
            return stream
        if usage := completion.get("usage"):
            usage_chunk = {**chunk, "choices": [], "usage": usage}
            await stream.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
        await stream.write(b"data: [DONE]\n\n")
        return stream

    async def __aenter__(self) -> Self:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1"
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.runner.cleanup()


def make(provider: Provider, harness, **client) -> vf.Agent:
    agent = vf.Agent(
        vf.AgentConfig(
            model="scripted",
            client={
                "type": "eval",
                "base_url": provider.url,
                "api_key_var": "SCRIPTED_KEY",
                **client,
            },
            runtime={"type": "subprocess"},
            harness={"id": "null"},
        )
    )
    agent.harness = harness(agent.harness.config)
    return agent


def task() -> NotesTask:
    return NotesTask(vf.TaskData(idx=0, prompt="start"))


class StreamHarness(ScriptHarness):
    """One streamed model call, read to its end."""

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(data)
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        try:
            stream = await client.chat.completions.create(
                model=ctx.model,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOLS,
                stream=True,
            )
            async for _ in stream:
                pass
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


class DroppedReaderHarness(ScriptHarness):
    """One streamed model call whose connection drops once the interception has
    committed the stream (a tunnel cut), then the SDK's retry of the same request,
    read to its end: the shape of a harness behind a tunnel in a long provider queue."""

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(data)
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        request = {
            "model": ctx.model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "stream": True,
        }
        try:
            dropped = await client.chat.completions.create(**request)
            await dropped.close()
            stream = await client.chat.completions.create(
                **request, extra_headers={server.RETRY_COUNT_HEADER: "1"}
            )
            async for _ in stream:
                pass
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


class CountingHarness(ScriptHarness):
    """One model call through an SDK that would retry 3 times, counting the requests
    it sends to the interception."""

    sent = 0

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        async def count(_request: httpx.Request) -> None:
            CountingHarness.sent += 1

        client = AsyncOpenAI(
            base_url=endpoint,
            api_key=secret,
            max_retries=3,
            http_client=httpx.AsyncClient(event_hooks={"request": [count]}),
        )
        try:
            await client.chat.completions.create(
                model=ctx.model, messages=[{"role": "user", "content": "hi"}]
            )
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


def assert_committed_once(trace: vf.Trace, sampled: int) -> None:
    """The retried calls committed each turn once: `sampled` replies, no repeats."""
    assert len([node for node in trace.nodes if node.sampled]) == sampled
    messages = [json.dumps(m.model_dump(), sort_keys=True) for m in trace.messages]
    assert len(messages) == len(set(messages)), "a message was committed twice"


@pytest.mark.parametrize(
    "failure",
    [502, 503, 500, 429, "reset"],
    ids=["cloudflare-502", "503-reset-before-headers", "500", "429", "dropped"],
)
async def test_a_transient_failure_is_retried_in_place(failure):
    """Every call of the rollout fails twice, then succeeds: the rollout is unharmed."""
    async with Provider(failure, failure, "ok", failure, "ok", "ok") as provider:
        agent = make(provider, ScriptHarness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 6
    assert_committed_once(trace, sampled=3)
    # Each failed exchange keeps its own record; none is a committed turn.
    assert [call.node is None for call in trace.calls] == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert all(call.error is not None for call in trace.calls if call.node is None)


@pytest.mark.parametrize(
    "harness", [ScriptHarness, StreamHarness], ids=["json", "stream"]
)
async def test_an_empty_turn_with_tool_call_markup_in_its_reasoning_is_retried(harness):
    """A reply whose tool calls the provider failed to parse (no content, no tool calls,
    raw tool-call markup in the reasoning) is a provider failure, retried in place,
    not an empty final turn that ends the run."""
    async with Provider("leak", "ok", "ok", "ok") as provider:
        agent = make(provider, harness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert trace.calls[0].node is None
    assert trace.calls[0].error is not None and trace.calls[0].error.status_code == 502
    assert len(provider.requests) == (4 if harness is ScriptHarness else 2)


@pytest.mark.parametrize(
    "harness", [ScriptHarness, StreamHarness], ids=["json", "stream"]
)
async def test_an_empty_turn_whose_usage_counts_unsent_output_is_retried(harness):
    """A reply with no content and no tool calls whose usage counts output past its
    reasoning (the provider generated the tool call and sent none of it) is a provider
    failure, retried in place, not an empty final turn that ends the run."""
    async with Provider("dropped", "ok", "ok", "ok") as provider:
        agent = make(provider, harness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert trace.calls[0].node is None
    assert trace.calls[0].error is not None and trace.calls[0].error.status_code == 502
    assert "never arrived" in trace.calls[0].error.message
    assert len(provider.requests) == (4 if harness is ScriptHarness else 2)


@pytest.mark.parametrize(
    "harness", [ScriptHarness, StreamHarness], ids=["json", "stream"]
)
async def test_an_empty_turn_of_one_token_is_retried(harness):
    """A reply of one end-of-turn token with no content, tool calls or reasoning is a
    provider failure, retried in place."""
    async with Provider("eos", "ok", "ok", "ok") as provider:
        agent = make(provider, harness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert trace.calls[0].node is None
    assert trace.calls[0].error is not None and trace.calls[0].error.status_code == 502
    assert len(provider.requests) == (4 if harness is ScriptHarness else 2)


async def test_an_empty_turn_after_reasoning_is_kept():
    """A model that reasons and then ends with an empty reply (its usage counting only
    an end-of-turn token past the reasoning) ends its run: nothing is retried."""
    async with Provider("done") as provider:
        agent = make(provider, StreamHarness)
        trace = await agent.run(task())
    assert len(provider.requests) == 1
    assert all(call.error is None for call in trace.calls)


@pytest.mark.parametrize(
    "harness", [ScriptHarness, StreamHarness], ids=["json", "stream"]
)
async def test_a_stream_of_keepalives_only_is_retried(harness, monkeypatch):
    """A provider stream that sends only keepalive comments (which keep the read timeout
    from firing) for longer than KEEPALIVE_ONLY_SECONDS is a stalled turn, retried in
    place rather than waited on."""
    monkeypatch.setattr(server, "KEEPALIVE_ONLY_SECONDS", 0.3)
    async with Provider("keepalives", "ok", "ok", "ok") as provider:
        agent = make(provider, harness, timeout={"read": 5.0})
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert trace.calls[0].node is None
    assert "only keepalives" in trace.calls[0].error.message
    assert len(provider.requests) == (4 if harness is ScriptHarness else 2)


async def test_a_read_timeout_is_retried_in_place():
    async with Provider((1.0, None), "ok", "ok", "ok") as provider:
        agent = make(provider, ScriptHarness, timeout={"read": 0.3})
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert trace.calls[0].error is not None
    assert trace.calls[0].error.status_code == 504
    assert trace.calls[0].error.message == "ReadTimeout"
    assert_committed_once(trace, sampled=3)


async def test_a_stream_cut_short_is_retried_in_place():
    async with Provider("truncate", "ok") as provider:
        agent = make(provider, StreamHarness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 2
    assert_committed_once(trace, sampled=1)


async def test_a_committed_stream_carries_its_retries(monkeypatch):
    """Past the keepalive grace the harness's stream is committed (200 + keepalives);
    the retries happen behind it and the success is served on it."""
    monkeypatch.setattr(server, "KEEPALIVE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(server, "KEEPALIVE_INTERVAL_SECONDS", 0.01)
    async with Provider((0.1, 502), (0.1, 503), (0.1, "reset"), "ok") as provider:
        agent = make(provider, StreamHarness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 4
    assert_committed_once(trace, sampled=1)


async def test_a_failure_after_the_reader_dropped_is_retried(monkeypatch):
    """The failure that lost the campaign's rollouts: the harness's connection drops
    while the call waits in the provider's queue, the SDK's retry coalesces onto the
    call in flight, and the call then fails upstream. It is retried in place and the
    coalesced retry gets the success, not the failure as a final SSE error."""
    monkeypatch.setattr(server, "KEEPALIVE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(server, "KEEPALIVE_INTERVAL_SECONDS", 0.01)
    async with Provider((0.5, 502), "ok") as provider:
        agent = make(provider, DroppedReaderHarness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 2, "the coalesced retry sent nothing upstream"
    assert_committed_once(trace, sampled=1)


@pytest.mark.parametrize("status", [400, 401, 404, 413, 422])
async def test_a_non_transient_failure_is_final_at_once(status):
    """An invalid request, a context overflow, auth: sending it again fails the same."""
    CountingHarness.sent = 0
    async with Provider(status, "ok") as provider:
        agent = make(provider, CountingHarness)
        trace = await agent.run(task())
    assert not trace.ok
    assert trace.last_error is not None
    assert trace.last_error.type == "ProviderError"
    assert trace.last_error.status_code == status
    assert len(provider.requests) == 1
    assert CountingHarness.sent == 1, "the SDK did not retry the final failure"


async def test_retries_stop_when_the_budget_runs_out():
    """A provider that keeps failing is retried for the budget, then the failure is
    final: recorded on the trace and marked so the harness SDK does not retry it."""
    CountingHarness.sent = 0
    async with Provider(*[503] * 1000) as provider:
        agent = make(provider, CountingHarness, retry={"budget": 0.5})
        started = time.monotonic()
        trace = await agent.run(task())
        elapsed = time.monotonic() - started
    assert not trace.ok
    assert trace.last_error is not None
    assert trace.last_error.status_code == 503
    assert RESET_BEFORE_HEADERS in trace.last_error.message
    assert len(provider.requests) > 2
    # No attempt starts past the budget (plus the first attempt's own latency).
    assert provider.times[-1] - provider.times[0] < 0.6
    assert elapsed < 5
    assert CountingHarness.sent == 1, "the SDK did not retry the final failure"
    assert len(trace.calls) == len(provider.requests)
    assert all(call.error is not None for call in trace.calls)


async def test_a_zero_budget_retries_nothing():
    async with Provider(502, "ok") as provider:
        agent = make(provider, CountingHarness, retry={"budget": 0})
        trace = await agent.run(task())
    assert not trace.ok
    assert len(provider.requests) == 1


async def test_retry_after_is_honoured():
    """The provider's `Retry-After` (0.3 s on these 429s) outlasts the backoff."""
    async with Provider(429, "ok", "ok", "ok") as provider:
        agent = make(provider, ScriptHarness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert provider.times[1] - provider.times[0] >= 0.3


class GatedTask(NotesTask):
    """Its request hook fails on the tool-call gate, after the reply committed."""

    @vf.intercept
    async def gate(self, request: vf.Request) -> None:
        if request.messages[-1].role == "tool":
            raise RuntimeError("the gate broke")


async def test_a_hook_failing_after_the_commit_is_not_sampled_again(monkeypatch):
    """Only the provider exchange is retried: a hook that fails after the reply
    committed fails the call, since sampling again would commit another turn."""
    monkeypatch.setattr(server, "KEEPALIVE_GRACE_SECONDS", 0.0)  # commit at once
    async with Provider() as provider:
        agent = make(provider, StreamHarness)
        trace = await agent.run(GatedTask(vf.TaskData(idx=0, prompt="start")))
    assert not trace.ok
    assert "the gate broke" in trace.errors[-1].message
    assert len(provider.requests) == 1
    assert len([node for node in trace.nodes if node.sampled]) == 1


async def test_a_long_stream_cut_short_restarts_the_budget():
    """The provider was serving the turn when it cut the stream: that time is no failing
    time, so a long turn's regeneration does not spend the budget (on 2026-10-20 GLM cut
    hour-long reasoning streams, and the second cut was already past a 900 s budget). The
    cut is recorded with how long the stream ran and how much of it arrived."""
    async with Provider((0.4, "cut"), (0.4, "truncate"), "ok") as provider:
        agent = make(provider, StreamHarness, retry={"budget": 0.3})
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 3
    cut = trace.calls[0].error
    assert cut is not None and cut.message.startswith(
        "upstream stream ended before its terminal event (after "
    )
    assert "1 events" in cut.message and "last event:" in cut.message
    assert_committed_once(trace, sampled=1)


async def test_a_provider_cutting_every_stream_still_gives_up(monkeypatch):
    """Cut-offs restart the budget only `MAX_STREAM_CUT_OFFS` times: a provider that cuts
    every stream still makes the failure final."""
    monkeypatch.setattr(server, "RETRY_BACKOFF_SECONDS", 0.001)
    async with Provider(*[(0.3, "cut")] * 50) as provider:
        agent = make(provider, StreamHarness, retry={"budget": 0.2})
        trace = await agent.run(task())
    assert not trace.ok
    assert trace.last_error is not None and "terminal event" in trace.last_error.message
    # the first cut, then MAX_STREAM_CUT_OFFS - 1 that restart the budget, then one past it
    assert len(provider.requests) == server.MAX_STREAM_CUT_OFFS + 1


async def test_an_error_event_inside_a_stream_is_retried_in_place():
    """Nemotron's gateway fails a turn mid-stream with an SSE error event ("Service error.
    Please retry shortly.") and then [DONE]. Forwarded, the harness's SDK raised it and the
    rollout died after hours; it is a cut, sampled again in place."""
    async with Provider("error-event", "error-event", "ok") as provider:
        agent = make(provider, StreamHarness, retry={"budget": 0.3})
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 3
    error = trace.calls[0].error
    assert error is not None and "Service error. Please retry shortly." in error.message
    assert_committed_once(trace, sampled=1)


async def test_a_non_streaming_call_streams_from_the_provider():
    """A harness asking for a plain JSON completion: the provider is asked to stream
    (with usage), and the harness gets the assembled completion."""
    async with Provider("ok", "ok", "ok") as provider:
        agent = make(provider, ScriptHarness)
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert all(r.get("stream") is True for r in provider.requests)
    assert all(r["stream_options"]["include_usage"] for r in provider.requests)
    assert_committed_once(trace, sampled=3)


async def test_a_long_generation_is_not_abandoned_by_the_read_timeout():
    """A turn that takes longer than the read timeout, streaming all along: it completes
    in one exchange instead of timing out and being sent again (the abandoned one would
    keep running at the provider, holding a concurrent slot)."""
    async with Provider("slow", "ok", "ok") as provider:
        agent = make(provider, ScriptHarness, timeout={"read": 0.3})
        trace = await agent.run(task())
    assert trace.ok, trace.errors
    assert len(provider.requests) == 3
    assert all(call.error is None for call in trace.calls)
