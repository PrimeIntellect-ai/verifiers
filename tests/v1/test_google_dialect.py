import json

import httpx
import pytest

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects.google import (
    GenerateContentResponse,
    GoogleGenerateContentDialect,
)
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.types import SamplingConfig


def test_google_generate_content_request_parsing():
    dialect = GoogleGenerateContentDialect()
    prompt, tools = dialect.parse_request(
        {
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "What is here?"},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "aW1hZ2U=",
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "I should inspect it.", "thought": True},
                        {
                            "functionCall": {
                                "id": "call-1",
                                "name": "inspect",
                                "args": {"detail": "high"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "inspect",
                                "response": {"result": "a diagram"},
                            }
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "inspect",
                            "description": "Inspect an image.",
                            "parameters": {
                                "type": "object",
                                "properties": {"detail": {"type": "string"}},
                            },
                        }
                    ]
                }
            ],
        }
    )

    assert [message.role for message in prompt] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert prompt[0].content == "Be concise."
    assert prompt[1].content[1].image_url.url == "data:image/png;base64,aW1hZ2U="
    assert prompt[2].reasoning_content == "I should inspect it."
    assert prompt[2].tool_calls[0].id == "call-1"
    assert prompt[2].tool_calls[0].arguments == '{"detail": "high"}'
    assert prompt[3].tool_call_id == "call-1"
    assert json.loads(prompt[3].content) == {"result": "a diagram"}
    assert tools[0].name == "inspect"


def test_google_generate_content_response_parsing():
    dialect = GoogleGenerateContentDialect()
    response = dialect.parse_response(
        GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Thinking.", "thought": True},
                                {"text": "Done."},
                                {
                                    "functionCall": {
                                        "id": "call-2",
                                        "name": "save",
                                        "args": {"value": 3},
                                    }
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 4,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 16,
                },
                "modelVersion": "gemini-3.5-flash",
                "responseId": "response-1",
            }
        )
    )

    assert response.id == "response-1"
    assert response.model == "gemini-3.5-flash"
    assert response.message.content == "Done."
    assert response.message.reasoning_content == "Thinking."
    assert response.message.tool_calls[0].name == "save"
    assert response.finish_reason == "tool_calls"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 6


def test_google_generate_content_stream_parsing():
    dialect = GoogleGenerateContentDialect()
    raw = b"\r\n\r\n".join(
        [
            b'data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Hel"}]}}],"responseId":"response-2","modelVersion":"gemini-flash"}',
            b'data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"lo"},{"text":"Plan","thought":true}]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":3,"totalTokenCount":8}}',
        ]
    )

    response = dialect.parse_stream(raw)

    assert response.message.content == "Hello"
    assert response.message.reasoning_content == "Plan"
    assert response.finish_reason == "length"
    assert response.usage.completion_tokens == 5


def test_google_generate_content_routes_auth_and_sampling():
    dialect = GoogleGenerateContentDialect()
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {
            "temperature": 1,
            "topP": 0.9,
            "maxOutputTokens": 20,
            "candidateCount": 1,
        },
    }

    assert dialect.auth_headers("provider-key") == {"x-goog-api-key": "provider-key"}
    assert dialect.secret({"x-goog-api-key": "rollout-secret"}) == "rollout-secret"
    assert (
        dialect.upstream_route("gemini-3.5-flash")
        == "/v1beta/models/gemini-3.5-flash:generateContent"
    )
    assert (
        dialect.upstream_route("tunedModels/123", stream=True)
        == "/v1beta/tunedModels/123:streamGenerateContent?alt=sse"
    )
    assert dialect.streaming(body, "/v1beta/models/gemini:streamGenerateContent")
    assert not dialect.streaming(body, "/v1beta/models/gemini:generateContent")

    steered = dialect.apply_overrides(
        body,
        "ignored-in-body",
        SamplingConfig(temperature=0.2, top_p=0.4, max_tokens=50),
    )

    assert steered["generationConfig"] == {
        "candidateCount": 1,
        "temperature": 0.2,
        "topP": 0.4,
        "maxOutputTokens": 50,
    }


@pytest.mark.asyncio
async def test_google_generate_content_transport_and_interception_routes():
    dialect = GoogleGenerateContentDialect()
    client = EvalClient("https://generativelanguage.googleapis.com", "provider-key")
    body = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
    url, headers, upstream = client._upstream(
        dialect,
        body,
        "gemini-3.5-flash",
        SamplingConfig(max_tokens=25),
        None,
    )
    stream_url, _, _ = client._upstream(
        dialect,
        body,
        "gemini-3.5-flash",
        SamplingConfig(),
        None,
        stream=True,
    )
    await client.close()

    assert (
        url == "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash:generateContent"
    )
    assert (
        stream_url == "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash:streamGenerateContent?alt=sse"
    )
    assert headers == {"x-goog-api-key": "provider-key"}
    assert upstream["generationConfig"] == {"maxOutputTokens": 25}

    async with InterceptionServer() as server:
        async with httpx.AsyncClient() as http:
            for method in ("generateContent", "streamGenerateContent"):
                response = await http.post(
                    f"http://127.0.0.1:{server.port}/v1beta/models/"
                    f"gemini-3.5-flash:{method}",
                    headers={"x-goog-api-key": "unknown-rollout"},
                    json=body,
                )
                assert response.status_code == 401
                assert response.json()["error"]["status"] == "INVALID_ARGUMENT"
