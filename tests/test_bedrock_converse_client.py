from unittest.mock import MagicMock, patch

import pytest

boto3 = pytest.importorskip("boto3", reason="boto3 not installed (verifiers[bedrock] extra)")

from verifiers.types import (  # noqa: E402
    AssistantMessage,
    ClientConfig,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


def _make_client():
    """Create a BedrockConverseClient with a mocked boto3 client."""
    from verifiers.clients.bedrock_converse_client import BedrockConverseClient

    config = ClientConfig(
        client_type="bedrock_converse",
        api_base_url="region:us-east-1",
        api_key_var="AWS_PROFILE",
    )

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_bedrock = MagicMock()
        mock_session.client.return_value = mock_bedrock

        client = BedrockConverseClient(config)

    return client, mock_bedrock


def _make_bedrock_response(*, content: str = "Hello", stop_reason: str = "end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": content}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "ResponseMetadata": {"RequestId": "req-123"},
    }


@pytest.mark.asyncio
async def test_to_native_prompt_basic():
    client, _ = _make_client()
    messages = [
        SystemMessage(content="You are a doctor."),
        UserMessage(content="What is hypertension?"),
    ]
    prompt, kwargs = await client.to_native_prompt(messages)

    assert prompt["system"] == [{"text": "You are a doctor."}]
    assert len(prompt["messages"]) == 1
    assert prompt["messages"][0] == {"role": "user", "content": [{"text": "What is hypertension?"}]}


@pytest.mark.asyncio
async def test_to_native_prompt_multi_turn():
    client, _ = _make_client()
    messages = [
        UserMessage(content="Hi"),
        AssistantMessage(content="Hello!"),
        UserMessage(content="What is a CBC?"),
    ]
    prompt, _ = await client.to_native_prompt(messages)

    assert len(prompt["messages"]) == 3
    assert prompt["messages"][0]["role"] == "user"
    assert prompt["messages"][1]["role"] == "assistant"
    assert prompt["messages"][2]["role"] == "user"


@pytest.mark.asyncio
async def test_to_native_prompt_tool_message():
    client, _ = _make_client()
    messages = [
        UserMessage(content="Calculate BMI"),
        ToolMessage(content="BMI is 24.5", tool_call_id="call-1"),
    ]
    prompt, _ = await client.to_native_prompt(messages)

    assert prompt["messages"][1]["role"] == "user"
    tool_result = prompt["messages"][1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "call-1"
    assert tool_result["content"] == [{"text": "BMI is 24.5"}]


@pytest.mark.asyncio
async def test_from_native_response_basic():
    client, _ = _make_client()
    response = _make_bedrock_response(content="High blood pressure.", stop_reason="end_turn")
    result = await client.from_native_response(response)

    assert result.message.content == "High blood pressure."
    assert result.message.finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.message.is_truncated is False


@pytest.mark.asyncio
async def test_from_native_response_max_tokens():
    client, _ = _make_client()
    response = _make_bedrock_response(content="Truncated...", stop_reason="max_tokens")
    result = await client.from_native_response(response)

    assert result.message.finish_reason == "length"
    assert result.message.is_truncated is True


@pytest.mark.asyncio
async def test_from_native_response_tool_use():
    client, _ = _make_client()
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Let me calculate that."},
                    {"toolUse": {"toolUseId": "tc-1", "name": "calculate_bmi", "input": {"weight": 70, "height": 1.75}}},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 20, "outputTokens": 15},
        "ResponseMetadata": {"RequestId": "req-456"},
    }
    result = await client.from_native_response(response)

    assert result.message.content == "Let me calculate that."
    assert result.message.finish_reason == "tool_calls"
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].name == "calculate_bmi"
    assert result.message.tool_calls[0].id == "tc-1"


@pytest.mark.asyncio
async def test_to_native_tool():
    from verifiers.types import Tool

    client, _ = _make_client()
    tool = Tool(
        name="calculate_bmi",
        description="Calculate BMI from weight and height",
        parameters={"type": "object", "properties": {"weight": {"type": "number"}, "height": {"type": "number"}}},
    )
    native = await client.to_native_tool(tool)

    assert native["toolSpec"]["name"] == "calculate_bmi"
    assert native["toolSpec"]["description"] == "Calculate BMI from weight and height"
    assert "json" in native["toolSpec"]["inputSchema"]


@pytest.mark.asyncio
async def test_get_native_response_calls_converse():
    client, mock_bedrock = _make_client()
    mock_bedrock.converse.return_value = _make_bedrock_response()

    prompt = {"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}
    sampling_args = {"temperature": 0.7, "max_tokens": 100}

    await client.get_native_response(prompt, "us.amazon.nova-pro-v1:0", sampling_args)

    mock_bedrock.converse.assert_called_once()
    call_kwargs = mock_bedrock.converse.call_args[1]
    assert call_kwargs["modelId"] == "us.amazon.nova-pro-v1:0"
    assert call_kwargs["inferenceConfig"]["temperature"] == 0.7
    assert call_kwargs["inferenceConfig"]["maxTokens"] == 100
