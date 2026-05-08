from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI
import pytest

from verifiers.clients import AnthropicMessagesClient, OpenAIResponsesClient
from verifiers.types import ClientConfig, Response, ResponseMessage, ToolCall
from verifiers.utils.interception_utils import serialize_intercept_response
from verifiers.v1.runtime import Runtime
from verifiers.v1.state import State
from verifiers.v1.utils.endpoint_utils import (
    Endpoint,
    normalize_endpoint_api,
    normalize_endpoint_prompt,
)


def test_runtime_records_client_config_protocol():
    runtime = Runtime()
    state = State({"runtime": {}})

    runtime.bind_model_client(
        state,
        ClientConfig(
            client_type="anthropic_messages",
            api_base_url="https://api.anthropic.com",
            api_key_var="ANTHROPIC_API_KEY",
        ),
    )

    assert state["runtime"]["client_type"] == "anthropic_messages"
    assert isinstance(runtime.model_client(state), AnthropicMessagesClient)


def test_runtime_preserves_concrete_client_config_protocol():
    runtime = Runtime()
    state = State({"runtime": {}})
    client = OpenAIResponsesClient(
        ClientConfig(client_type="openai_responses", api_key_var="OPENAI_API_KEY")
    )

    runtime.bind_model_client(state, client)

    assert state["runtime"]["client_type"] == "openai_responses"
    assert runtime.model_client(state) is client


def test_endpoint_client_protocol_accepts_explicit_api_surface():
    endpoint = Endpoint(port=9999)
    root = "http://127.0.0.1:9999/rollout/test"
    state = State(
        {
            "runtime": {"client_type": "openai_responses"},
            "endpoint_root_url": root,
            "endpoint_base_url": f"{root}/v1",
        }
    )

    openai = endpoint.client(state)
    openai_sync = endpoint.client(state, api="chat", sync=True)
    completions = endpoint.client(state, api="openai_completions")
    responses = endpoint.client(state, api="openai_responses")
    anthropic = endpoint.client(state, api="anthropic_messages")
    anthropic_sync = endpoint.client(state, api="messages", sync=True)

    assert isinstance(openai, AsyncOpenAI)
    assert isinstance(openai_sync, OpenAI)
    assert isinstance(completions, AsyncOpenAI)
    assert isinstance(responses, AsyncOpenAI)
    assert isinstance(anthropic, AsyncAnthropic)
    assert isinstance(anthropic_sync, Anthropic)


def test_endpoint_config_uses_endpoint_client_type_names():
    endpoint = Endpoint(port=9999, secret="test-secret")
    root = "http://127.0.0.1:9999/rollout/test"
    state = State(
        {
            "runtime": {"model": "test-model"},
            "endpoint_root_url": root,
            "endpoint_base_url": f"{root}/v1",
        }
    )

    openai_config = endpoint.config(state, api="responses")
    anthropic_config = endpoint.config(state, api="messages")

    assert openai_config == {
        "model": "test-model",
        "api_key": "test-secret",
        "base_url": f"{root}/v1",
        "api_base": f"{root}/v1",
        "api_client_type": "openai_responses",
    }
    assert anthropic_config == {
        "model": "test-model",
        "api_key": "test-secret",
        "base_url": root,
        "api_client_type": "anthropic_messages",
    }


@pytest.mark.parametrize(
    ("alias", "api"),
    [
        ("chat", "chat_completions"),
        ("chat_completions", "chat_completions"),
        ("openai_chat_completions", "chat_completions"),
        ("completions", "completions"),
        ("openai_completions", "completions"),
        ("responses", "responses"),
        ("openai_responses", "responses"),
        ("messages", "messages"),
        ("anthropic_messages", "messages"),
    ],
)
def test_endpoint_api_aliases_match_endpoint_config_type(alias, api):
    assert normalize_endpoint_api(alias) == api


@pytest.mark.parametrize(
    "api",
    [
        "openai",
        "anthropic",
        "completion",
        "openai_chat_completions_token",
        "renderer",
        "nemorl_chat_completions",
    ],
)
def test_endpoint_api_rejects_unsupported_client_types(api):
    with pytest.raises(ValueError):
        normalize_endpoint_api(api)


def test_openai_completions_endpoint_prompt_normalizes_text_prompt():
    messages = normalize_endpoint_prompt(
        {"protocol": "openai_completions", "prompt": "Complete this sentence"}
    )

    assert messages[0].role == "text"
    assert messages[0].content == "Complete this sentence"


def test_anthropic_endpoint_prompt_normalizes_tool_messages():
    messages = normalize_endpoint_prompt(
        {
            "protocol": "anthropic_messages",
            "system": "system text",
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "search",
                            "input": {"query": "x"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "answer",
                        }
                    ],
                },
            ],
        }
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["tool_calls"][0]["name"] == "search"
    assert messages[3]["role"] == "tool"


def test_openai_responses_serialization_includes_function_calls():
    response = Response(
        id="resp_1",
        created=123,
        model="m",
        usage=None,
        message=ResponseMessage(
            content=None,
            finish_reason="tool_calls",
            is_truncated=False,
            tool_calls=[
                ToolCall(id="call_1", name="search", arguments='{"query": "x"}')
            ],
        ),
    )

    payload = serialize_intercept_response(response, protocol="openai_responses")

    assert payload["object"] == "response"
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["call_id"] == "call_1"


def test_openai_completions_serialization_returns_text_completion_shape():
    response = Response(
        id="cmpl_1",
        created=123,
        model="m",
        usage=None,
        message=ResponseMessage(
            content="done",
            finish_reason="stop",
            is_truncated=False,
        ),
    )

    payload = serialize_intercept_response(response, protocol="openai_completions")

    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == "done"
