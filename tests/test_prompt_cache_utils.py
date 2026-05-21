from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import ClientConfig, Response, ResponseMessage
from verifiers.utils.prompt_cache_utils import (
    apply_prompt_cache_to_request,
    endpoint_origin,
    uses_official_anthropic_messages,
)
from verifiers.utils.save_utils import state_to_output
from verifiers.utils.usage_utils import extract_usage_token_details


class RecordingClient(Client):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.request = {}

    def setup_client(self, config):
        return object()

    async def to_native_tool(self, tool):
        return tool

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(
        self, prompt, model, sampling_args, tools=None, **kwargs
    ):
        self.request = {
            "prompt": prompt,
            "model": model,
            "sampling_args": sampling_args,
            "tools": tools,
            "kwargs": kwargs,
        }
        return object()

    async def raise_from_native_response(self, response):
        _ = response

    async def from_native_response(self, response):
        _ = response
        return Response(
            id="resp",
            created=0,
            model="model",
            usage=None,
            message=ResponseMessage(
                content="ok",
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    async def close(self) -> None:
        pass


def test_endpoint_origin_normalizes_urls():
    assert (
        endpoint_origin("https://api.anthropic.com/v1") == "https://api.anthropic.com"
    )
    assert endpoint_origin("https://api.anthropic.com:443/v1") == (
        "https://api.anthropic.com"
    )
    assert endpoint_origin("http://localhost:8080/v1") == "http://localhost:8080"
    assert endpoint_origin("http://[::1]:8080/v1") == "http://[::1]:8080"


def test_official_anthropic_messages_endpoint_is_cache_control_target():
    assert uses_official_anthropic_messages(
        ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com",
        )
    )
    assert not uses_official_anthropic_messages(
        ClientConfig(
            client_type="openai_chat_completions",
            api_base_url="https://api.anthropic.com",
        )
    )
    assert not uses_official_anthropic_messages(
        ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.pinference.ai/api/v1",
        )
    )


def test_anthropic_request_adds_top_level_cache_control():
    native_prompt, native_tools, sampling_args, extra_kwargs = (
        apply_prompt_cache_to_request(
            config=ClientConfig(
                client_type="anthropic_messages",
                api_base_url="https://api.anthropic.com",
            ),
            model="claude-sonnet-4-5",
            native_prompt=[{"role": "user", "content": "question"}],
            native_tools=None,
            sampling_args={"max_tokens": 16},
            extra_kwargs={},
        )
    )

    assert native_prompt == [{"role": "user", "content": "question"}]
    assert native_tools is None
    assert sampling_args == {"max_tokens": 16}
    assert extra_kwargs["cache_control"] == {"type": "ephemeral"}


def test_anthropic_request_preserves_sampling_args_cache_control():
    _, _, sampling_args, extra_kwargs = apply_prompt_cache_to_request(
        config=ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com",
        ),
        model="claude-sonnet-4-5",
        native_prompt=[],
        native_tools=None,
        sampling_args={
            "max_tokens": 16,
            "cache_control": {"type": "custom"},
        },
        extra_kwargs={},
    )

    assert sampling_args == {
        "max_tokens": 16,
        "cache_control": {"type": "custom"},
    }
    assert extra_kwargs == {}


def test_openai_request_does_not_mutate_request():
    native_prompt, native_tools, sampling_args, extra_kwargs = (
        apply_prompt_cache_to_request(
            config=ClientConfig(
                client_type="openai_chat_completions",
                api_base_url="https://api.openai.com/v1",
            ),
            model="gpt-5.4-mini",
            native_prompt=[{"role": "user", "content": "question"}],
            native_tools=None,
            sampling_args={"max_tokens": 16},
            extra_kwargs={},
        )
    )

    assert native_prompt == [{"role": "user", "content": "question"}]
    assert native_tools is None
    assert sampling_args == {"max_tokens": 16}
    assert extra_kwargs == {}


def test_non_official_anthropic_endpoint_does_not_add_cache_control():
    _, _, sampling_args, extra_kwargs = apply_prompt_cache_to_request(
        config=ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.pinference.ai/api/v1",
        ),
        model="claude-sonnet-4-5",
        native_prompt=[],
        native_tools=None,
        sampling_args={"max_tokens": 16},
        extra_kwargs={},
    )

    assert sampling_args == {"max_tokens": 16}
    assert extra_kwargs == {}


@pytest.mark.asyncio
async def test_client_request_hook_applies_anthropic_cache_control():
    client = RecordingClient(
        ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com",
        )
    )

    await client.get_response(
        prompt=[],
        model="claude-sonnet-4-5",
        sampling_args={"max_tokens": 16},
    )

    assert client.request["kwargs"]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_openai_usage_splits_cached_input_tokens():
    client = OpenAIChatCompletionsClient(object())
    message = SimpleNamespace(
        content="ok",
        tool_calls=None,
        model_dump=lambda: {},
    )
    native_response = SimpleNamespace(
        id="resp",
        created=0,
        model="gpt-5.4-mini",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_write_tokens=10,
            ),
        ),
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason="stop",
            )
        ],
    )

    response = await client.from_native_response(native_response)

    assert response.usage is not None
    assert response.usage.prompt_tokens == 20
    assert response.usage.cached_input_tokens == 80
    assert response.usage.total_tokens == 25


@pytest.mark.asyncio
async def test_anthropic_usage_splits_cache_read_and_write_tokens():
    client = AnthropicMessagesClient(object())
    native_response = SimpleNamespace(
        id="resp",
        model="claude-sonnet-4-5",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=7,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=10,
        ),
    )

    response = await client.from_native_response(native_response)

    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.cached_input_tokens == 80
    assert response.usage.total_tokens == 22


def test_native_anthropic_cache_creation_counts_as_uncached_input():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=7,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=10,
        )
    )

    assert extract_usage_token_details(response) == {
        "input_tokens": 15,
        "output_tokens": 7,
        "cached_input_tokens": 80,
    }


def test_serialized_response_usage_counts_cache_details():
    response = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 10,
            },
        }
    }

    assert extract_usage_token_details(response) == {
        "input_tokens": 20,
        "output_tokens": 7,
        "cached_input_tokens": 80,
    }


def test_serialized_responses_usage_counts_input_token_cache_details():
    response = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 7,
            "input_tokens_details": {
                "cached_tokens": 80,
            },
        }
    }

    assert extract_usage_token_details(response) == {
        "input_tokens": 20,
        "output_tokens": 7,
        "cached_input_tokens": 80,
    }


def test_state_output_fallback_reads_serialized_trajectory_usage():
    task = vf.Task(
        {
            "example_id": 0,
            "prompt": [{"role": "user", "content": "q"}],
        }
    ).freeze()
    state = vf.State.for_task(task)
    state["trajectory"] = [
        {
            "response": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 80},
                }
            }
        }
    ]

    output = state_to_output(state)

    assert output["token_usage"]["input_tokens"] == 20.0
    assert output["token_usage"]["cached_input_tokens"] == 80.0
    assert output["token_usage"]["output_tokens"] == 7.0
