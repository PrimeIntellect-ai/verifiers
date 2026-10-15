"""The chat-completions helper retries an interrupted stream from scratch and folds the
streamed reasoning fields of the successful attempt into one assistant message."""

import json

import httpx
import pytest
from openai import APIError, AsyncOpenAI

from verifiers.v1.harnesses.utils.core import chat


def chunk(text: str, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning": text,
                    "reasoning_content": text,
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "index": 0,
                            "id": "r1",
                            "format": "unknown",
                            "signature": "sig",
                            "text": text,
                        }
                    ],
                },
                "finish_reason": finish_reason,
            }
        ],
    }


@pytest.mark.parametrize(
    "interruption",
    [None, "disconnect", "timeout", "eof", "retryable-error", "terminal-error"],
)
@pytest.mark.parametrize(
    ("headers", "expected_name", "expected_namespace"),
    [
        (
            [("call_", "get_", "wea"), ("123", "weather", "ther")],
            "get_weather",
            "weather",
        ),
        ([("call_123", "get_weather", "weather")] * 2, "get_weather", "weather"),
        ([("call_123", "echo", "fs"), (None, "echo", "fs")], "echoecho", "fsfs"),
    ],
)
async def test_chat_preserves_streamed_reasoning(
    interruption, headers, expected_name, expected_namespace
):
    events = [chunk("Plan: "), chunk("call ls", "tool_calls")]
    for event, (call_id, name, namespace), arguments in zip(
        events, headers, ['{"city":', '"Paris"}']
    ):
        event["choices"][0]["delta"]["tool_calls"] = [
            {
                "index": 0,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "namespace": namespace,
                    "arguments": arguments,
                },
            }
        ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    requests = []
    streams = []

    class CompletionStream(httpx.AsyncByteStream):
        closed = False

        def __init__(self, request, interrupted):
            self.request = request
            self.interrupted = interrupted

        async def __aiter__(self):
            if self.interrupted:
                yield f"data: {json.dumps(chunk('discard this attempt'))}\n\n".encode()
                if interruption == "disconnect":
                    raise httpx.ReadError("connection reset", request=self.request)
                if interruption == "timeout":
                    raise httpx.ReadTimeout("read timed out", request=self.request)
                if interruption in ("retryable-error", "terminal-error"):
                    error = {
                        "error": {
                            "message": "upstream failed",
                            "retryable": interruption == "retryable-error",
                        }
                    }
                    yield f"data: {json.dumps(error)}\n\n".encode()
                return
            yield content.encode()

        async def aclose(self):
            self.closed = True

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        stream = CompletionStream(request, interruption and len(requests) == 1)
        streams.append(stream)
        return httpx.Response(
            200,
            stream=stream,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
        if interruption == "terminal-error":
            with pytest.raises(APIError, match="upstream failed"):
                await chat(client, "test-model", [], [])
        else:
            completion = await chat(client, "test-model", [], [])

    retried = interruption is not None and interruption != "terminal-error"
    assert len(requests) == (2 if retried else 1)
    assert [r.headers["x-stainless-retry-count"] for r in requests] == (
        ["0", "1"] if retried else ["0"]
    )
    assert all(r.content == requests[0].content for r in requests)
    assert all(stream.closed for stream in streams)
    if interruption == "terminal-error":
        return
    message = completion.choices[0].message.model_dump(exclude_none=True)
    assert message["role"] == "assistant"
    assert message["reasoning"] == "Plan: call ls"
    assert message["reasoning_content"] == "Plan: call ls"
    assert message["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "index": 0,
            "id": "r1",
            "format": "unknown",
            "signature": "sig",
            "text": "Plan: call ls",
        }
    ]
    (tool_call,) = message["tool_calls"]
    assert tool_call["id"] == "call_123"
    assert tool_call["function"] == {
        "name": expected_name,
        "namespace": expected_namespace,
        "arguments": '{"city":"Paris"}',
    }
