"""The chat-completions helper retries an interrupted stream from scratch and folds the
streamed reasoning fields of the successful attempt into one assistant message."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

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


@pytest.mark.parametrize("interruption", [None, "disconnect", "timeout", "eof"])
async def test_chat_preserves_streamed_reasoning(interruption):
    events = [chunk("Plan: "), chunk("call ls", "stop")]
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
        completion = await chat(client, "test-model", [], [])

    assert len(requests) == (2 if interruption else 1)
    assert [r.headers["x-stainless-retry-count"] for r in requests] == (
        ["0", "1"] if interruption else ["0"]
    )
    assert all(r.content == requests[0].content for r in requests)
    assert all(stream.closed for stream in streams)
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
