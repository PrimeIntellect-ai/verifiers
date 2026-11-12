"""Each dialect imposes the eval's sampling in its native request shape, replacing the
harness's own values, and parses the same shape back for the trace."""

import httpx
import openai
import pytest

from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect

SAMPLING = SamplingConfig(temperature=0.3, top_p=0.9, reasoning_effort="low")


@pytest.mark.parametrize(
    "dialect,body,effort",
    [
        (
            ChatDialect(),
            {"temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"},
            lambda body: body["reasoning_effort"],
        ),
        (
            ResponsesDialect(),
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "reasoning": {"effort": "high", "summary": "detailed"},
            },
            lambda body: body["reasoning"]["effort"],
        ),
        (
            AnthropicDialect(),
            {"temperature": 1.0, "top_p": 1.0, "output_config": {"effort": "high"}},
            lambda body: body["output_config"]["effort"],
        ),
    ],
    ids=["chat", "responses", "anthropic"],
)
def test_apply_overrides_sampling(dialect, body, effort):
    sent = dialect.apply_overrides(body, "m", SAMPLING)
    assert (sent["temperature"], sent["top_p"], effort(sent)) == (0.3, 0.9, "low")
    parsed = dialect.parse_sampling(sent)
    assert (parsed.temperature, parsed.top_p, parsed.reasoning_effort) == (
        0.3,
        0.9,
        "low",
    )


def test_apply_overrides_keeps_other_reasoning_fields():
    body = {"reasoning": {"effort": "high", "summary": "detailed"}}
    sent = ResponsesDialect().apply_overrides(body, "m", SAMPLING)
    assert sent["reasoning"] == {"effort": "low", "summary": "detailed"}


def conversation(turns: int) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "s"}]
    for turn in range(turns):
        messages.append({"role": "user", "content": f"q{turn}"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{turn}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"a": 1}'},
                    }
                ],
            }
        )
        # Tool results arrive unnamed: the parse names them from their call.
        messages.append(
            {"role": "tool", "tool_call_id": f"call_{turn}", "content": f"r{turn}"}
        )
    return messages


def test_chat_parse_request_reuses_the_last_requests_equal_prefix():
    """A harness re-sends its whole conversation every turn: the messages it begins with,
    equal as sent to the last request's, are that request's parse; the rest parse as before,
    tool results named from calls anywhere in the conversation."""
    dialect = ChatDialect()
    first = conversation(2)
    last = (first, dialect.parse_request({"messages": first}).messages)
    grown = conversation(3)
    request = dialect.parse_request({"messages": grown}, last)
    assert request == dialect.parse_request({"messages": grown})
    assert all(a is b for a, b in zip(request.messages, last[1]))
    assert request.messages[-1].name == "lookup"
    # An edited earlier message parses from there on, whatever follows it.
    edited = [*grown[:2], {"role": "user", "content": "q0, edited"}, *grown[3:]]
    request = dialect.parse_request({"messages": edited}, last)
    assert request == dialect.parse_request({"messages": edited})
    assert request.messages[0] is last[1][0]
    assert all(a is not b for a, b in zip(request.messages[2:], last[1][2:]))


def test_message_hash_is_kept_with_the_message():
    from verifiers.v1 import graph

    messages = ChatDialect().parse_request({"messages": conversation(2)}).messages
    for message in messages:
        assert graph.message_hash(message) == graph._message_hash(message)
        assert graph.message_hash(message) == graph._message_hash(message)
    assert all(id(message) in graph._HASHES for message in messages)
    keys = [id(message) for message in messages]
    del message, messages
    assert not any(key in graph._HASHES for key in keys)


@pytest.mark.parametrize(
    "status,kind",
    [(502, "server_error"), (429, "rate_limit_error"), (400, "invalid_request_error")],
)
@pytest.mark.parametrize(
    "dialect", [ChatDialect(), ResponsesDialect()], ids=["chat", "responses"]
)
async def test_committed_stream_error_keeps_its_status(dialect, status, kind):
    """A turn that fails after its stream is committed (HTTP 200 sent, keepalives flowing)
    reaches the harness's OpenAI SDK as an error carrying the status it would have had, so
    the harness retries a 5xx or 429 and not a 4xx."""
    error = dialect.error_body("upstream failed", status_code=status)
    events = dialect.stream_keepalive(True) + dialect.stream_error(
        error, status_code=status
    )

    def serve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=events
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(serve))
    async with openai.AsyncOpenAI(
        api_key="k", base_url="http://interception/v1", http_client=http_client
    ) as client:
        with pytest.raises(openai.APIError) as raised:
            if isinstance(dialect, ChatDialect):
                stream = await client.chat.completions.create(
                    model="m", messages=[{"role": "user", "content": "q"}], stream=True
                )
            else:
                stream = await client.responses.create(
                    model="m", input="q", stream=True
                )
            async for _ in stream:
                pass
    assert (raised.value.code, raised.value.type) == (status, kind)
