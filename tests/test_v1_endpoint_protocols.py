from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verifiers.clients import AnthropicMessagesClient, OpenAIResponsesClient
from verifiers.types import ClientConfig, Response, ResponseMessage, ToolCall
from verifiers.utils.interception_utils import serialize_intercept_response
from verifiers.v1.runtime import Runtime
from verifiers.v1.state import State
from verifiers.v1.utils.endpoint_utils import Endpoint, normalize_endpoint_prompt


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


def test_endpoint_client_protocol_uses_state_client_type():
    endpoint = Endpoint(port=9999)
    root = "http://127.0.0.1:9999/rollout/test"

    anthropic = endpoint.client(
        State(
            {
                "runtime": {"client_type": "anthropic_messages"},
                "endpoint_root_url": root,
                "endpoint_base_url": f"{root}/v1",
            }
        )
    )
    openai = endpoint.client(
        State(
            {
                "runtime": {"client_type": "openai_responses"},
                "endpoint_root_url": root,
                "endpoint_base_url": f"{root}/v1",
            }
        )
    )

    assert isinstance(anthropic, AsyncAnthropic)
    assert isinstance(openai, AsyncOpenAI)


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
