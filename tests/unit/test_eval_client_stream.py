"""With `stream` set, the eval client streams each call and assembles the reply the program
asked for: the same response an unstreamed call returns, failures mid-stream included."""

import json

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import eval as eval_client
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.errors import ProviderError

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    }
]
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Calling it.",
                "reasoning": "The user wants an echo.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": USAGE,
}


def chunk(delta: dict | None = None, finish_reason=None, **extra) -> dict:
    choices = (
        []
        if delta is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": choices,
        **extra,
    }


# With `logprobs` on, vLLM sends a chunk every step: an empty delta while a tool call is generated.
LOGPROB_STEP = chunk({})
LOGPROB_STEP["choices"][0]["logprobs"] = {
    "content": [{"token": "x", "logprob": -0.1, "bytes": [120], "top_logprobs": []}]
}

# The same reply as COMPLETION, streamed the way vLLM sends it: reasoning and content deltas,
# a logprob-only step, a tool call whose arguments arrive in pieces, then a usage chunk
# without choices.
CHUNKS = [
    chunk({"role": "assistant", "content": ""}),
    chunk({"reasoning": "The user "}),
    chunk({"reasoning": "wants an echo."}),
    chunk({"content": "Calling "}),
    chunk({"content": "it."}),
    LOGPROB_STEP,
    chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": ""},
                }
            ]
        }
    ),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"text": '}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]}),
    chunk({}, finish_reason="tool_calls"),
    chunk(usage=USAGE),
]


def sse(events: list[dict | str]) -> bytes:
    lines = [
        event if isinstance(event, str) else f"data: {json.dumps(event)}\n\n"
        for event in events
    ]
    return "".join(lines).encode()


class Endpoint:
    """A provider that records request bodies. It answers an unstreamed call with
    COMPLETION and a streamed one with `stream` in small pieces (so SSE events straddle
    reads), then raises `error` if given."""

    def __init__(self, stream: bytes, *, error: Exception | None = None):
        self.stream = stream
        self.error = error
        self.bodies: list[dict] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if not body.get("stream"):
            return httpx.Response(200, json=COMPLETION)

        async def pieces():
            for start in range(0, len(self.stream), 7):
                yield self.stream[start : start + 7]
            if self.error is not None:
                raise self.error

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=pieces()
        )


@pytest.fixture
def connect(monkeypatch):
    def connect(endpoint: Endpoint, *, stream: bool = True) -> eval_client.EvalClient:
        monkeypatch.setattr(
            eval_client,
            "build_async_httpx",
            lambda config: httpx.AsyncClient(
                transport=httpx.MockTransport(endpoint.handle)
            ),
        )
        return eval_client.EvalClient(
            vf.EvalClientConfig(
                base_url="http://model.test/v1",
                api_key_var="TEST_API_KEY",
                stream=stream,
            )
        )

    return connect


async def call(client: eval_client.EvalClient, body: dict = BODY):
    return await client.get_response(ChatDialect(), body, SamplingConfig())


async def test_stream_assembles_the_unstreamed_response(connect):
    endpoint = Endpoint(sse([": keepalive\n\n", *CHUNKS, "data: [DONE]\n\n"]))
    streamed = await call(connect(endpoint))
    unstreamed = await call(connect(endpoint, stream=False))

    sent, plain = endpoint.bodies
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert "stream" not in plain
    assert streamed.message == unstreamed.message
    assert streamed.message.reasoning_content == "The user wants an echo."
    assert [(call.name, call.arguments) for call in streamed.message.tool_calls] == [
        ("echo", '{"text": "hi"}')
    ]
    assert (streamed.finish_reason, streamed.usage) == (
        unstreamed.finish_reason,
        unstreamed.usage,
    )
    assert streamed.usage.completion_tokens == 7
    # The program is served a chat.completion, as if it had not been streamed.
    assert streamed.raw["object"] == "chat.completion"
    assert (
        streamed.raw["choices"][0]["message"]["tool_calls"]
        == COMPLETION["choices"][0]["message"]["tool_calls"]
    )


async def test_tool_choice_none_drops_streamed_tool_calls(connect):
    # vLLM's stream still returns the calls its unstreamed reply drops for `tool_choice: "none"`.
    endpoint = Endpoint(sse([*CHUNKS, "data: [DONE]\n\n"]))
    response = await call(connect(endpoint), {**BODY, "tool_choice": "none"})

    assert endpoint.bodies[0]["tool_choice"] == "none"
    assert response.message.content == "Calling it."
    assert response.message.tool_calls is None
    assert response.finish_reason == "stop"
    assert response.raw["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in response.raw["choices"][0]["message"]


@pytest.mark.parametrize(
    "error,status",
    [
        (httpx.ReadTimeout("read timed out"), 504),
        (httpx.RemoteProtocolError("peer closed connection"), 502),
        (None, 502),  # ended cleanly before its terminal event
    ],
    ids=["read-timeout", "connection-dropped", "truncated"],
)
async def test_stream_cut_mid_reply_is_a_retryable_provider_error(
    connect, error, status
):
    endpoint = Endpoint(sse(CHUNKS[:5]), error=error)
    with pytest.raises(ProviderError) as raised:
        await call(connect(endpoint))
    assert raised.value.status_code == status
