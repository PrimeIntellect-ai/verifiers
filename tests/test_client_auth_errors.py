import httpx
import pytest
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError

from verifiers.clients.anthropic.anthropic_clients import AnthropicMessagesClient
from verifiers.clients.openai.openai_clients import (
    OAIChatCompletionsClient,
    OAICompletionsClient,
)
from verifiers.errors import OverlongPromptError
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


# ---------------------------------------------------------------------------
# Overlong-prompt detection
# ---------------------------------------------------------------------------


def _make_anthropic_overlong_error(message: str) -> AnthropicBadRequestError:
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        text=message,
    )
    return AnthropicBadRequestError(message, response=response, body=None)


def _make_openai_overlong_error(message: str) -> OpenAIBadRequestError:
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        text=message,
    )
    return OpenAIBadRequestError(message, response=response, body=None)


class _OverlongAnthropicMessages:
    def __init__(self, message: str) -> None:
        self._message = message

    async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise _make_anthropic_overlong_error(self._message)


class _OverlongAnthropicClient:
    def __init__(self, message: str) -> None:
        self.messages = _OverlongAnthropicMessages(message)


class _OverlongOpenAIChatCompletions:
    def __init__(self, message: str) -> None:
        self._message = message

    async def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise _make_openai_overlong_error(self._message)


class _OverlongOpenAIChatClient:
    class _Chat:
        def __init__(self, message: str) -> None:
            self.completions = _OverlongOpenAIChatCompletions(message)

    def __init__(self, message: str) -> None:
        self.chat = self._Chat(message)


@pytest.mark.parametrize(
    "error_message",
    [
        "prompt is too long: 300000 tokens > 200000 maximum",
        "input_length and max_tokens exceed context limit: 199211+20000 > 200000",
        "too many total text bytes: 17929476 >> 16000000",
        "Your input is too long. Please reduce.",
    ],
)
@pytest.mark.asyncio
async def test_anthropic_overlong_prompt_raises_overlong_error(error_message: str):
    client = AnthropicMessagesClient(_OverlongAnthropicClient(error_message))

    with pytest.raises(OverlongPromptError):
        await client.get_response(
            prompt=[UserMessage(content="test prompt")],
            model="claude-test",
            sampling_args={"max_tokens": 16},
        )


@pytest.mark.asyncio
async def test_anthropic_non_overlong_bad_request_not_converted():
    """A BadRequestError that is NOT about context length should propagate as-is (→ ModelError)."""
    from verifiers.errors import ModelError

    client = AnthropicMessagesClient(
        _OverlongAnthropicClient("invalid parameter: temperature must be >= 0")
    )

    with pytest.raises(ModelError):
        await client.get_response(
            prompt=[UserMessage(content="test prompt")],
            model="claude-test",
            sampling_args={"max_tokens": 16},
        )


@pytest.mark.parametrize(
    "error_message",
    [
        "This model's maximum context length is 128000 tokens",
        "prompt_too_long",
        "context length exceeded",
    ],
)
@pytest.mark.asyncio
async def test_openai_overlong_prompt_raises_overlong_error(error_message: str):
    client = OAIChatCompletionsClient(_OverlongOpenAIChatClient(error_message))

    with pytest.raises(OverlongPromptError):
        await client.get_response(
            prompt=[UserMessage(content="test prompt")],
            model="gpt-test",
            sampling_args={},
        )
