from unittest.mock import AsyncMock, Mock

import pytest
import verifiers.v1 as vf
from anthropic.types import Message as AnthropicMessage
from google.genai import types as google_types
from openai.types.chat import ChatCompletion
from openai.types.responses import Response as OpenAIResponse
from pydantic import TypeAdapter

from verifiers.v1 import graph
from verifiers.v1.clients.anthropic import (
    AnthropicMessagesClient,
    content_to_wire as anthropic_content,
    messages_to_wire as anthropic_messages,
)
from verifiers.v1.clients.anthropic import (
    response_from_wire as anthropic_response,
)
from verifiers.v1.clients.google import (
    GoogleResponsesClient,
    content_to_wire as google_content,
)
from verifiers.v1.clients.google import messages_to_wire as google_messages
from verifiers.v1.clients.google import response_from_wire as google_response
from verifiers.v1.clients.openai import (
    OpenAIChatCompletionsClient,
    content_to_wire as chat_content,
    message_to_wire as chat_message,
)
from verifiers.v1.clients.openai_responses import (
    OpenAIResponsesClient,
    content_to_wire as responses_content,
)
from verifiers.v1.clients.openai_responses import message_to_wire as responses_message
from verifiers.v1.clients.openai_responses import (
    response_from_wire as responses_response,
)
from verifiers.v1.types import content_to_parts


class AsyncItems:
    def __init__(self, items=(), final=None):
        self.items = iter(items)
        self.final = final

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get_final_message(self):
        return self.final

    async def get_final_completion(self):
        return self.final


def round_trip(response: vf.Response, prompt: vf.Messages) -> vf.AssistantMessage:
    trace = vf.Trace(task=vf.Task(idx=0, instruction="use a tool"))
    graph.add_turn(trace, prompt, response)
    return vf.Trace.model_validate(trace.to_wire()).assistant_messages[0]


def image(url: str, detail: str | None = None) -> vf.ImageUrlContentPart:
    return vf.ImageUrlContentPart(
        image_url=vf.ImageUrlSource.model_validate({"url": url, "detail": detail})
    )


def test_image_detail_round_trips_to_openai_clients():
    content = content_to_parts(
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "high"},
            }
        ]
    )

    assert isinstance(content, list)
    assert chat_content(content)[0]["image_url"]["detail"] == "high"
    assert responses_content(content)[0]["detail"] == "high"
    assert chat_content([image("data:image/png;base64,aW1hZ2U=")])[0]["image_url"][
        "url"
    ].startswith("data:image/png;base64,")
    assert responses_content([image("data:image/png;base64,aW1hZ2U=")])[0][
        "image_url"
    ].startswith("data:image/png;base64,")
    assert (
        responses_content([image("https://example.com/image.png", "original")])[0][
            "detail"
        ]
        == "original"
    )
    with pytest.raises(ValueError, match="does not support image detail"):
        chat_content([image("https://example.com/image.png", "original")])


def test_openai_system_image_support_matches_native_apis():
    system = vf.SystemMessage(content=[vf.TextContentPart(text="System")])
    assert chat_message(system)["content"] == [{"type": "text", "text": "System"}]

    system = vf.SystemMessage(content=[image("data:image/png;base64,aW1hZ2U=")])
    with pytest.raises(ValueError, match="system messages do not support images"):
        chat_message(system)
    assert responses_message(system)[0]["content"][0]["type"] == "input_image"


def test_anthropic_uses_native_url_and_base64_image_sources():
    content = anthropic_content(
        [
            image("https://example.com/image.png"),
            image("data:IMAGE/PNG;charset=utf-8;BASE64,aW1hZ2U="),
        ]
    )

    assert isinstance(content, list)
    assert content[0]["source"] == {
        "type": "url",
        "url": "https://example.com/image.png",
    }
    assert content[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "aW1hZ2U=",
    }
    with pytest.raises(ValueError, match="must be base64 encoded"):
        anthropic_content([image("data:image/png,image")])


def test_google_uses_inline_images():
    parts = google_content(
        [
            image(
                "data:IMAGE/PNG;charset=utf-8;BASE64,aW1hZ2U=",
                "high",
            )
        ]
    )

    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "image/png"
    assert parts[0].inline_data.data == b"image"
    assert parts[0].media_resolution is not None
    assert parts[0].media_resolution.level == (
        google_types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
    )
    with pytest.raises(ValueError, match="must use data URIs"):
        google_content([image("https://example.com/image.png")])
    with pytest.raises(ValueError, match="must be base64 encoded"):
        google_content([image("data:image/png,image")])


def test_openai_responses_preserves_native_output():
    output = [
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Need weather."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "weather",
            "arguments": '{"city":"Berlin"}',
        },
    ]
    response = responses_response(
        OpenAIResponse.model_validate(
            {
                "id": "resp_1",
                "created_at": 0,
                "model": "gpt-test",
                "object": "response",
                "status": "completed",
                "output": output,
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
    )

    assistant = round_trip(response, [vf.UserMessage(content="Weather?")])

    assert responses_message(assistant) == output
    assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt_token_ids", "expected_prompt_ids"),
    [([1, 2], [1, 2]), (None, [])],
)
async def test_openai_chat_preserves_tokens_and_logprobs(
    prompt_token_ids, expected_prompt_ids
):
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl_1",
            "created": 0,
            "model": "vllm-test",
            "object": "chat.completion",
            "prompt_token_ids": prompt_token_ids,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": [3, 4],
                    "message": {
                        "role": "assistant",
                        "content": "Hi",
                        "reasoning": "Think",
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "text": "Think",
                                "format": "anthropic-claude-v1",
                                "index": 0,
                            }
                        ],
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "H",
                                "bytes": [72],
                                "logprob": -0.1,
                                "top_logprobs": [],
                            },
                            {
                                "token": "i",
                                "bytes": [105],
                                "logprob": -0.2,
                                "top_logprobs": [],
                            },
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }
    )
    openai = Mock()
    openai.chat.completions.create = AsyncMock(return_value=completion)
    client = OpenAIChatCompletionsClient(openai)

    response = await client.get_response(
        [vf.UserMessage(content="Hello")],
        "vllm-test",
        vf.SamplingConfig.model_validate(
            {
                "logprobs": True,
                "extra_body": {"return_token_ids": True},
            }
        ),
    )

    await_args = openai.chat.completions.create.await_args
    assert await_args is not None
    assert await_args.kwargs["logprobs"] is True
    assert await_args.kwargs["extra_body"] == {"return_token_ids": True}
    assert response.message.reasoning_content == "Think"
    assert response.message.provider_state is not None
    assert chat_message(response.message)["reasoning_details"] == (
        response.message.provider_state
    )
    assert response.tokens is not None
    assert response.tokens.prompt_ids == expected_prompt_ids
    assert response.tokens.completion_ids == [3, 4]
    assert response.tokens.completion_logprobs == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_openai_chat_passes_reasoning_options_through():
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl_1",
            "created": 0,
            "model": "anthropic/claude-haiku-4.5",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi"},
                }
            ],
        }
    )
    openai = Mock()
    openai.chat.completions.create = AsyncMock(return_value=completion)
    client = OpenAIChatCompletionsClient(openai)

    await client.get_response(
        [vf.UserMessage(content="Hello")],
        "anthropic/claude-opus-4.6",
        vf.SamplingConfig.model_validate(
            {
                "reasoning_effort": "high",
                "verbosity": "max",
                "extra_body": {"reasoning": {"enabled": True}},
            }
        ),
    )

    request = openai.chat.completions.create.await_args.kwargs
    assert request["reasoning_effort"] == "high"
    assert request["verbosity"] == "max"
    assert request["extra_body"] == {"reasoning": {"enabled": True}}


@pytest.mark.asyncio
async def test_openai_chat_aggregates_stream():
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl_1",
            "created": 0,
            "model": "vllm-test",
            "object": "chat.completion",
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "token_ids": [3, 4],
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "Think more",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Berlin"}',
                                },
                            }
                        ],
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "a",
                                "bytes": [97],
                                "logprob": -0.1,
                                "top_logprobs": [],
                            },
                            {
                                "token": "b",
                                "bytes": [98],
                                "logprob": -0.2,
                                "top_logprobs": [],
                            },
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }
    )
    openai = Mock()
    openai.chat.completions.stream = Mock(return_value=AsyncItems(final=completion))
    openai.chat.completions.create = AsyncMock()
    client = OpenAIChatCompletionsClient(openai)

    response = await client.get_response(
        [vf.UserMessage(content="Weather?")],
        "vllm-test",
        vf.SamplingConfig.model_validate(
            {
                "stream": True,
                "logprobs": True,
                "extra_body": {"return_token_ids": True},
            }
        ),
    )

    request = openai.chat.completions.stream.call_args.kwargs
    assert "stream" not in request
    assert request["stream_options"]["include_usage"] is True
    assert response.message.reasoning_content == "Think more"
    assert response.message.tool_calls == [
        vf.ToolCall(
            id="call_1",
            name="weather",
            arguments='{"city":"Berlin"}',
        )
    ]
    assert response.finish_reason == "tool_calls"
    assert response.usage == vf.Usage(prompt_tokens=2, completion_tokens=2)
    assert response.tokens is not None
    assert response.tokens.prompt_ids == [1, 2]
    assert response.tokens.completion_ids == [3, 4]
    assert response.tokens.completion_logprobs == [-0.1, -0.2]
    openai.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_responses_aggregates_stream():
    final = OpenAIResponse.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-test",
            "object": "response",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {"type": "output_text", "text": "Hi", "annotations": []}
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )
    event = Mock(type="response.incomplete", response=final)
    openai = Mock()
    openai.responses.create = AsyncMock(return_value=AsyncItems([event]))
    client = OpenAIResponsesClient(openai)

    response = await client.get_response(
        [vf.UserMessage(content="Hello")],
        "gpt-test",
        vf.SamplingConfig.model_validate({"stream": True}),
    )

    assert openai.responses.create.await_args.kwargs["stream"] is True
    assert response.message.content == "Hi"
    assert response.finish_reason == "length"
    assert response.usage == vf.Usage(prompt_tokens=1, completion_tokens=1)


def test_anthropic_preserves_thinking_blocks():
    thinking = {
        "type": "thinking",
        "thinking": "Need weather.",
        "signature": "signed-thinking",
    }
    response = anthropic_response(
        AnthropicMessage.model_validate(
            {
                "id": "msg_1",
                "model": "claude-test",
                "role": "assistant",
                "type": "message",
                "stop_reason": "tool_use",
                "content": [
                    thinking,
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "weather",
                        "input": {"city": "Berlin"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
    )

    assistant = round_trip(response, [vf.UserMessage(content="Weather?")])
    prompt = anthropic_messages(
        [
            assistant,
            vf.ToolMessage(tool_call_id="call_1", content='{"temp":20}'),
        ]
    )

    assert list(prompt[0]["content"])[0] == thinking
    assert list(prompt[1]["content"])[0]["tool_use_id"] == "call_1"


@pytest.mark.asyncio
async def test_anthropic_requires_max_tokens():
    anthropic = Mock()
    anthropic.messages.create = AsyncMock()
    client = AnthropicMessagesClient(anthropic)

    with pytest.raises(ValueError, match="requires max_tokens"):
        await client.get_response(
            [vf.UserMessage(content="Hello")],
            "claude-test",
            vf.SamplingConfig(),
        )


@pytest.mark.asyncio
async def test_anthropic_passes_native_options():
    anthropic = Mock()
    anthropic.messages.create = AsyncMock(
        return_value=AnthropicMessage.model_validate(
            {
                "id": "msg_1",
                "model": "claude-test",
                "role": "assistant",
                "type": "message",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    )
    client = AnthropicMessagesClient(anthropic)

    await client.get_response(
        [
            vf.SystemMessage(content=[vf.TextContentPart(text="System")]),
            vf.UserMessage(content="Hello"),
        ],
        "claude-test",
        vf.SamplingConfig.model_validate(
            {
                "max_tokens": 100,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            }
        ),
    )

    await_args = anthropic.messages.create.await_args
    assert await_args is not None
    request = await_args.kwargs
    assert request["system"] == "System"
    assert request["max_tokens"] == 100
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}

    with pytest.raises(ValueError, match="system messages do not support images"):
        await client.get_response(
            [
                vf.SystemMessage(content=[image("data:image/png;base64,aW1hZ2U=")]),
                vf.UserMessage(content="Hello"),
            ],
            "claude-test",
            vf.SamplingConfig(max_tokens=100),
        )


@pytest.mark.asyncio
async def test_anthropic_aggregates_stream():
    final = AnthropicMessage.model_validate(
        {
            "id": "msg_1",
            "model": "claude-test",
            "role": "assistant",
            "type": "message",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    anthropic = Mock()
    anthropic.messages.stream = Mock(return_value=AsyncItems(final=final))
    anthropic.messages.create = AsyncMock()
    client = AnthropicMessagesClient(anthropic)

    response = await client.get_response(
        [vf.UserMessage(content="Hello")],
        "claude-test",
        vf.SamplingConfig.model_validate({"max_tokens": 100, "stream": True}),
    )

    request = anthropic.messages.stream.call_args.kwargs
    assert "stream" not in request
    assert response.message.content == "Hi"
    assert response.finish_reason == "stop"
    anthropic.messages.create.assert_not_awaited()


def test_google_preserves_thought_signatures():
    part = {
        "functionCall": {
            "id": "call_1",
            "name": "weather",
            "args": {"city": "Berlin"},
        },
        "thoughtSignature": "c2lnbmVkLXRob3VnaHQ=",
    }
    response = google_response(
        google_types.GenerateContentResponse.model_validate(
            {
                "responseId": "response_1",
                "modelVersion": "gemini-test",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": [part]},
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 15},
            }
        ),
        "gemini-test",
    )

    assistant = round_trip(response, [vf.UserMessage(content="Weather?")])
    prompt = google_messages(
        [
            assistant,
            vf.ToolMessage(tool_call_id="call_1", content='{"temp":20}'),
        ]
    )

    assert prompt[0].parts is not None
    assert (
        prompt[0].parts[0].model_dump(mode="json", by_alias=True, exclude_none=True)
        == part
    )
    assert prompt[1].parts is not None
    assert prompt[1].parts[0].function_response is not None
    assert prompt[1].parts[0].function_response.name == "weather"


@pytest.mark.asyncio
async def test_google_uses_native_config():
    response = google_types.GenerateContentResponse.model_validate(
        {
            "responseId": "response_1",
            "modelVersion": "gemini-test",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hi"}],
                    },
                }
            ],
        }
    )
    google = Mock()
    google.aio.models.generate_content = AsyncMock(return_value=response)
    client = GoogleResponsesClient(google)

    await client.get_response(
        [
            vf.SystemMessage(content="System"),
            vf.UserMessage(content="Hello"),
        ],
        "google/gemini-test",
        vf.SamplingConfig.model_validate(
            {
                "max_tokens": 100,
                "stop": "done",
                "top_k": 20,
                "logprobs": True,
                "top_logprobs": 5,
                "thinking_config": {"thinking_budget": 1000},
            }
        ),
        tools=[
            vf.Tool(
                name="weather",
                description="Get weather",
                parameters={"type": "object"},
            )
        ],
    )

    await_args = google.aio.models.generate_content.await_args
    assert await_args is not None
    request = await_args.kwargs
    assert request["model"] == "gemini-test"
    assert request["config"].max_output_tokens == 100
    assert request["config"].stop_sequences == ["done"]
    assert request["config"].top_k == 20
    assert request["config"].response_logprobs is True
    assert request["config"].logprobs == 5
    assert request["config"].system_instruction == "System"
    assert request["config"].thinking_config.thinking_budget == 1000
    assert request["config"].tools[0].function_declarations[0].name == "weather"

    await client.get_response(
        [
            vf.SystemMessage(content=[image("data:image/png;base64,aW1hZ2U=")]),
            vf.UserMessage(content="Hello"),
        ],
        "google/gemini-test",
        vf.SamplingConfig(),
    )

    system_instruction = google.aio.models.generate_content.await_args.kwargs[
        "config"
    ].system_instruction
    assert isinstance(system_instruction, list)
    assert system_instruction[0].inline_data is not None
    assert system_instruction[0].inline_data.data == b"image"


@pytest.mark.asyncio
async def test_google_aggregates_stream():
    chunks = [
        google_types.GenerateContentResponse.model_validate(
            {
                "responseId": "response_1",
                "modelVersion": "gemini-test",
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "Think",
                                    "thought": True,
                                    "thoughtSignature": "c2lnbmVkLXRob3VnaHQ=",
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        google_types.GenerateContentResponse.model_validate(
            {
                "responseId": "response_1",
                "modelVersion": "gemini-test",
                "candidates": [
                    {
                        "index": 0,
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hi"}],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "thoughtsTokenCount": 2,
                    "toolUsePromptTokenCount": 3,
                    "totalTokenCount": 7,
                },
            }
        ),
    ]
    google = Mock()
    google.aio.models.generate_content_stream = AsyncMock(
        return_value=AsyncItems(chunks)
    )
    google.aio.models.generate_content = AsyncMock()
    client = GoogleResponsesClient(google)

    response = await client.get_response(
        [vf.UserMessage(content="Hello")],
        "google/gemini-test",
        vf.SamplingConfig.model_validate({"stream": True}),
    )

    request = google.aio.models.generate_content_stream.await_args.kwargs
    assert "stream" not in request["config"].model_dump(exclude_none=True)
    assert response.message.reasoning_content == "Think"
    assert response.message.content == "Hi"
    assert response.message.provider_state is not None
    assert response.message.provider_state[0]["thoughtSignature"] == (
        "c2lnbmVkLXRob3VnaHQ="
    )
    assert response.finish_reason == "stop"
    assert response.usage == vf.Usage(prompt_tokens=1, completion_tokens=1)
    google.aio.models.generate_content.assert_not_awaited()


def test_client_config_protocols():
    adapter = TypeAdapter(vf.ClientConfig)
    for protocol in (
        "openai",
        "openai_responses",
        "anthropic_messages",
        "google_responses",
        "renderers",
    ):
        assert adapter.validate_python({"type": protocol}).type == protocol
