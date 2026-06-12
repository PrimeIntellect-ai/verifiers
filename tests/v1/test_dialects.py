"""The wire dialects and the interception server's relay/translate arms.

Codec tests round-trip each dialect's serializers through its own parsers (and validate
the serialized shapes against the provider SDK models, so a program's real SDK would
accept them). Server tests run a live `InterceptionServer` with stub clients: a typed
stub exercises the translate arm (any ingress dialect -> typed client -> native shape
back), a relay stub asserts byte-verbatim pass-through (JSON and SSE) on the matching
dialect.
"""

import json

import aiohttp
import verifiers.v1 as vf
from anthropic.types import Message as AnthropicSDKMessage
from openai.types.responses import Response as OpenAISDKResponse

from verifiers.v1.clients import RolloutContext
from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect, sse
from verifiers.v1.interception import InterceptionServer, RolloutSession
from verifiers.v1.trace import Trace

CHAT = ChatDialect()
ANTHROPIC = AnthropicDialect()
RESPONSES = ResponsesDialect()


def make_response(**overrides) -> vf.Response:
    fields = {
        "id": "resp_1",
        "created": 1,
        "model": "test-model",
        "message": vf.AssistantMessage(content="Hello!"),
        "finish_reason": "stop",
        "usage": vf.Usage(prompt_tokens=10, completion_tokens=5),
        **overrides,
    }
    return vf.Response(**fields)


TOOLED = make_response(
    message=vf.AssistantMessage(
        content="Using a tool.",
        reasoning_content="Let me check the weather.",
        tool_calls=[
            vf.ToolCall(id="call_1", name="weather", arguments='{"city":"Berlin"}')
        ],
    ),
    finish_reason="tool_calls",
)


# --- anthropic codec ------------------------------------------------------------


def test_anthropic_parse_request():
    body = {
        "model": "claude-x",
        "system": "Be terse.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U=",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hmm.", "signature": "sig"},
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "look",
                        "input": {"x": 1},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "a red square",
                    }
                ],
            },
        ],
        "tools": [
            {"name": "look", "description": "Look.", "input_schema": {"type": "object"}}
        ],
    }
    prompt, tools = ANTHROPIC.parse_request(body)
    system, user, assistant, tool = prompt
    assert isinstance(system, vf.SystemMessage) and system.content == "Be terse."
    assert isinstance(user, vf.UserMessage)
    assert user.content[1].image_url.url == "data:image/png;base64,aW1hZ2U="
    assert isinstance(assistant, vf.AssistantMessage)
    assert assistant.content == "Checking."
    assert assistant.reasoning_content == "Hmm."
    assert assistant.tool_calls[0].name == "look"
    assert json.loads(assistant.tool_calls[0].arguments) == {"x": 1}
    assert isinstance(tool, vf.ToolMessage) and tool.tool_call_id == "tu_1"
    assert tools[0].name == "look" and tools[0].parameters == {"type": "object"}


def test_anthropic_serialize_validates_and_reasoning_round_trips():
    wire = ANTHROPIC.serialize_response(TOOLED, "test-model")
    AnthropicSDKMessage.model_validate(wire)  # a real Anthropic SDK would accept it
    assert wire["stop_reason"] == "tool_use"
    # The program echoes our message back; parse_request must recover the reasoning
    # (the passback some models require) and the tool call.
    prompt, _ = ANTHROPIC.parse_request(
        {"messages": [{"role": "assistant", "content": wire["content"]}]}
    )
    (echoed,) = prompt
    assert echoed.reasoning_content == "Let me check the weather."
    assert echoed.content == "Using a tool."
    assert echoed.tool_calls[0].id == "call_1"


def test_anthropic_stream_round_trip():
    raw = ANTHROPIC.serialize_stream(TOOLED, "test-model")
    response = ANTHROPIC.parse_stream(raw)
    assert response.message.content == "Using a tool."
    assert response.message.reasoning_content == "Let me check the weather."
    assert response.message.tool_calls[0].arguments == '{"city": "Berlin"}'
    assert response.finish_reason == "tool_calls"
    assert response.usage.completion_tokens == 5


def test_anthropic_secret_carriers():
    assert ANTHROPIC.secret({"x-api-key": "s1"}) == "s1"
    assert ANTHROPIC.secret({"Authorization": "Bearer s2"}) == "s2"


def test_anthropic_count_tokens_estimate():
    estimate = ANTHROPIC.handle_aux(
        "/v1/messages/count_tokens",
        {"messages": [{"role": "user", "content": "word " * 100}]},
    )
    assert estimate["input_tokens"] > 50


# --- responses codec ------------------------------------------------------------


def test_responses_parse_request_groups_assistant_run():
    body = {
        "model": "gpt-x",
        "instructions": "Be terse.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Hmm."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        ],
        "tools": [
            {"type": "function", "name": "weather", "parameters": {}, "strict": True}
        ],
    }
    prompt, tools = RESPONSES.parse_request(body)
    system, user, assistant, tool = prompt
    assert isinstance(system, vf.SystemMessage) and system.content == "Be terse."
    assert isinstance(user, vf.UserMessage)
    assert isinstance(assistant, vf.AssistantMessage)
    assert assistant.reasoning_content == "Hmm."
    assert assistant.tool_calls[0].id == "call_1"
    assert (
        assistant.provider_state[0]["type"] == "reasoning"
    )  # native continuation state
    assert isinstance(tool, vf.ToolMessage) and tool.content == "sunny"
    assert tools[0].strict is True


def test_responses_parse_request_string_input():
    prompt, tools = RESPONSES.parse_request({"input": "hi"})
    assert isinstance(prompt[0], vf.UserMessage) and prompt[0].content == "hi"
    assert tools is None


def test_responses_serialize_validates_and_round_trips():
    wire = RESPONSES.serialize_response(TOOLED, "test-model")
    OpenAISDKResponse.model_validate(wire)  # a real OpenAI SDK would accept it
    kinds = [item["type"] for item in wire["output"]]
    assert kinds == ["reasoning", "message", "function_call"]
    response = RESPONSES.parse_response(wire)
    assert response.message.content == "Using a tool."
    assert response.message.tool_calls[0].name == "weather"
    assert response.finish_reason == "tool_calls"


def test_responses_stream_round_trip():
    raw = RESPONSES.serialize_stream(make_response(), "test-model")
    response = RESPONSES.parse_stream(raw)
    assert response.message.content == "Hello!"
    assert response.finish_reason == "stop"


# --- chat codec (stream paths; the JSON paths are covered in test_clients) -------


def test_chat_stream_round_trip():
    raw = CHAT.serialize_stream(TOOLED, "test-model")
    response = CHAT.parse_stream(raw)
    assert response.message.content == "Using a tool."
    assert response.message.reasoning_content == "Let me check the weather."
    assert response.message.tool_calls[0].arguments == '{"city":"Berlin"}'
    assert response.usage.completion_tokens == 5


def test_chat_parse_stream_accumulates_deltas():
    chunks = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}
            ],
        },
    ]
    raw = b"".join(sse(c) for c in chunks) + b"data: [DONE]\n\n"
    response = CHAT.parse_stream(raw)
    assert response.message.content == "Hello"
    assert response.finish_reason == "stop"


# --- the server's two arms --------------------------------------------------------


class TypedStub(Client):
    """Translate-arm client: no native dialect, returns a canned typed response."""

    def __init__(self, response: vf.Response) -> None:
        self.response = response
        self.seen_prompt = None

    async def get_response(self, prompt, model, sampling_args, tools=None):
        self.seen_prompt = prompt
        return self.response


class RelayStub(Client):
    """Relay-arm client: returns canned bytes, records what was relayed."""

    def __init__(self, payload: bytes, content_type: str = "application/json") -> None:
        self.payload = payload
        self.content_type = content_type
        self.relayed: tuple[bytes, str] | None = None

    async def get_response(self, prompt, model, sampling_args, tools=None):
        raise AssertionError("relay arm must not translate")

    async def relay(self, body: bytes, route: str) -> RelayReply:
        self.relayed = (body, route)

        async def chunks():
            yield self.payload

        return RelayReply(self.content_type, chunks())


class ChatRelayStub(RelayStub):
    dialect = "chat"


class AnthropicRelayStub(RelayStub):
    dialect = "anthropic"


def make_session(client: Client) -> RolloutSession:
    return RolloutSession(
        ctx=RolloutContext(
            client=client, model="test-model", sampling=vf.SamplingConfig()
        ),
        trace=Trace(task=vf.Task(idx=0, instruction="test")),
    )


async def serve(session: RolloutSession):
    server = InterceptionServer()
    await server.__aenter__()
    secret = server.register(session)
    return server, secret, f"http://127.0.0.1:{server.port}"


async def test_server_translates_anthropic_ingress():
    """An anthropic-dialect program against a non-anthropic client: typed translate,
    response handed back in anthropic shape, turn recorded."""
    client = TypedStub(make_response())
    session = make_session(client)
    server, secret, base = await serve(session)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/messages",
                json={
                    "model": "x",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"x-api-key": secret},
            )
            body = await resp.json()
        assert resp.status == 200
        assert body["type"] == "message"
        assert body["content"] == [{"type": "text", "text": "Hello!"}]
        assert isinstance(client.seen_prompt[0], vf.UserMessage)
        assert session.trace.num_turns == 1
    finally:
        await server.__aexit__(None, None, None)


async def test_server_relays_matching_dialect_bytes():
    """A chat-dialect program with a chat client: the request bytes reach the client
    verbatim, the canned upstream bytes come back verbatim, the turn is recorded."""
    upstream = json.dumps(
        CHAT.serialize_response(make_response(), "test-model")
    ).encode()
    client = ChatRelayStub(upstream)
    session = make_session(client)
    server, secret, base = await serve(session)
    request_body = json.dumps(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 7,
        }
    ).encode()
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/chat/completions",
                data=request_body,
                headers={
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                },
            )
            returned = await resp.read()
        assert resp.status == 200
        assert returned == upstream  # bytes out, untouched
        relayed_body, relayed_route = client.relayed
        assert (
            relayed_body == request_body
        )  # bytes in, untouched (incl. unknown `seed`)
        assert relayed_route == "/v1/chat/completions"
        assert session.trace.num_turns == 1
    finally:
        await server.__aexit__(None, None, None)


async def test_server_relays_stream_and_records_turn():
    """A streaming relay: SSE bytes pass through verbatim and the assembled final
    message lands on the trace."""
    payload = ANTHROPIC.serialize_stream(make_response(), "test-model")
    client = AnthropicRelayStub(payload, content_type="text/event-stream")
    session = make_session(client)
    server, secret, base = await serve(session)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/messages",
                json={
                    "model": "x",
                    "max_tokens": 16,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"x-api-key": secret},
            )
            returned = await resp.read()
        assert returned == payload
        assert session.trace.num_turns == 1
        assert session.trace.assistant_messages[0].content == "Hello!"
    finally:
        await server.__aexit__(None, None, None)


async def test_server_translate_streams_fake_sse():
    """A streaming program against a non-matching client: the typed response goes back
    as a minimal valid SSE stream in the program's dialect."""
    client = TypedStub(make_response())
    session = make_session(client)
    server, secret, base = await serve(session)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/messages",
                json={
                    "model": "x",
                    "max_tokens": 16,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"x-api-key": secret},
            )
            raw = await resp.read()
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        assert ANTHROPIC.parse_stream(raw).message.content == "Hello!"
        assert session.trace.num_turns == 1
    finally:
        await server.__aexit__(None, None, None)


async def test_server_aux_route_estimates_or_relays():
    """count_tokens: estimated locally on the translate arm, relayed on the relay arm."""
    session = make_session(TypedStub(make_response()))
    server, secret, base = await serve(session)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/messages/count_tokens",
                json={"messages": [{"role": "user", "content": "word " * 50}]},
                headers={"x-api-key": secret},
            )
            assert (await resp.json())["input_tokens"] > 20
    finally:
        await server.__aexit__(None, None, None)

    relay = AnthropicRelayStub(b'{"input_tokens": 123}')
    session = make_session(relay)
    server, secret, base = await serve(session)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/messages/count_tokens",
                json={"messages": []},
                headers={"x-api-key": secret},
            )
            assert (await resp.json())["input_tokens"] == 123
        assert relay.relayed[1] == "/v1/messages/count_tokens"
        assert session.trace.num_turns == 0  # aux calls are not model turns
    finally:
        await server.__aexit__(None, None, None)


async def test_server_refusal_uses_dialect_error_shape():
    """A turn-limit refusal halts an anthropic program with an Anthropic-shaped error."""
    from verifiers.v1.interception import RolloutLimits

    session = make_session(TypedStub(make_response()))
    session.limits = RolloutLimits(max_turns=0)
    server, secret, base = await serve(session)
    try:
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{base}/v1/messages",
                json={
                    "model": "x",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"x-api-key": secret},
            )
            body = await resp.json()
        assert resp.status == 400
        assert body["type"] == "error"
        assert "max_turns" in body["error"]["message"]
    finally:
        await server.__aexit__(None, None, None)
