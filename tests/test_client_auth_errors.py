import httpx
import pytest
from anthropic import AuthenticationError as AnthropicAuthenticationError
from openai import AuthenticationError as OpenAIAuthenticationError

from verifiers.clients.anthropic.anthropic_clients import AnthropicMessagesClient
from verifiers.clients.openai.openai_clients import OAICompletionsClient
from verifiers.types import TextMessage, UserMessage


def _make_openai_auth_error() -> OpenAIAuthenticationError:
    response = httpx.Response(
        status_code=401,
        request=httpx.Request("POST", "https://api.openai.com/v1/completions"),
        text="invalid api key",
    )
    return OpenAIAuthenticationError(
        "Authentication failed", response=response, body=None
    )


def _make_anthropic_auth_error() -> AnthropicAuthenticationError:
    response = httpx.Response(
        status_code=401,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        text="invalid api key",
    )
    return AnthropicAuthenticationError(
        "Authentication failed", response=response, body=None
    )


class _FailingOpenAICompletions:
    async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise _make_openai_auth_error()


class _FailingOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FailingOpenAICompletions()


class _FailingAnthropicMessages:
    async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise _make_anthropic_auth_error()


class _FailingAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FailingAnthropicMessages()


@pytest.mark.asyncio
async def test_openai_auth_error_not_wrapped_as_model_error():
    client = OAICompletionsClient(_FailingOpenAIClient())

    with pytest.raises(OpenAIAuthenticationError):
        await client.get_response(
            prompt=[TextMessage(content="test prompt")],
            model="gpt-test",
            sampling_args={},
        )


@pytest.mark.asyncio
async def test_anthropic_auth_error_not_wrapped_as_model_error():
    client = AnthropicMessagesClient(_FailingAnthropicClient())

    with pytest.raises(AnthropicAuthenticationError):
        await client.get_response(
            prompt=[UserMessage(content="test prompt")],
            model="claude-test",
            sampling_args={"max_tokens": 16},
        )
