import json

import pytest

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects.google import (
    GenerateContentResponse,
    GoogleGenerateContentDialect,
)
from verifiers.v1.types import SamplingConfig


def test_google_generate_content_request_parsing():
    prompt, tools = GoogleGenerateContentDialect().parse_request(
        {
            "systemInstruction": {"parts": [{"text": "Be concise."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Inspect this."}]},
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
                            "description": "Inspect an input.",
                            "parameters": {"type": "object"},
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
    assert prompt[1].content == "Inspect this."
    assert prompt[2].reasoning_content == "I should inspect it."
    assert prompt[2].tool_calls[0].id == "call-1"
    assert prompt[2].tool_calls[0].arguments == '{"detail": "high"}'
    assert json.loads(prompt[3].content) == {"result": "a diagram"}
    assert tools[0].name == "inspect"


def test_google_generate_content_response_parsing():
    response = GoogleGenerateContentDialect().parse_response(
        GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Thinking.", "thought": True},
                                {"text": "Done."},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "totalTokenCount": 16,
                },
                "modelVersion": "gemini-flash",
                "responseId": "response-1",
            }
        )
    )

    assert response.id == "response-1"
    assert response.message.content == "Done."
    assert response.message.reasoning_content == "Thinking."
    assert response.finish_reason == "stop"
    assert response.usage.completion_tokens == 6


@pytest.mark.asyncio
async def test_google_generate_content_transport_and_sampling():
    dialect = GoogleGenerateContentDialect()
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {"temperature": 1, "candidateCount": 1},
    }
    client = EvalClient("https://generativelanguage.googleapis.com", "provider-key")
    url, headers, upstream = client._upstream(
        dialect,
        body,
        "gemini-flash",
        SamplingConfig(temperature=0.2, max_tokens=50),
        None,
    )
    await client.close()

    assert (
        url == "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-flash:generateContent"
    )
    assert headers == {"x-goog-api-key": "provider-key"}
    assert upstream["generationConfig"] == {
        "candidateCount": 1,
        "temperature": 0.2,
        "maxOutputTokens": 50,
    }
