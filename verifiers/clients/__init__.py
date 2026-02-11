from __future__ import annotations

from typing import Literal, TypeVar, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.client import Client
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.clients.openai_completions_client import OpenAICompletionsClient
from verifiers.types import ClientConfig, ClientType, NativeClient
from verifiers.utils.client_utils import (
    _setup_anthropic_client_from_resolved,
    _setup_openai_client_from_resolved,
    resolve_client_config,
)


def get_provider(client_type: ClientType) -> Literal["openai", "anthropic"]:
    if client_type == "anthropic_messages":
        return "anthropic"
    return "openai"


def setup_native_client(config: ClientConfig) -> AsyncOpenAI | AsyncAnthropic:
    """Setup the appropriate async client based on config.client_type."""
    resolved_config = resolve_client_config(config)
    provider = get_provider(resolved_config.client_type)
    if provider == "openai":
        return _setup_openai_client_from_resolved(resolved_config)
    if provider == "anthropic":
        return _setup_anthropic_client_from_resolved(resolved_config)
    raise ValueError(f"Unsupported client type: {resolved_config.client_type}")


def resolve_client(client_or_config: Client | ClientConfig) -> Client:
    """Resolves a client or client config to a client."""
    if isinstance(client_or_config, Client):
        client = client_or_config
        return client
    elif isinstance(client_or_config, ClientConfig):
        resolved_config = resolve_client_config(client_or_config)
        native_client = setup_native_client(client_or_config)
        return resolve_native_client(native_client, resolved_config.client_type)
    else:
        raise ValueError(f"Unsupported client type: {type(client_or_config)}")


def resolve_native_client(
    native_client: NativeClient,
    client_type: ClientType,
) -> Client:
    """Resolves a native client (e.g. OpenAI or Anthropic) to vf.Client."""

    ExpectedNativeClientT = TypeVar("ExpectedNativeClientT", bound=NativeClient)

    def raise_on_invalid_client(
        client_type: ClientType,
        native_client: NativeClient,
        expected_native_client_cls: type[ExpectedNativeClientT],
    ) -> ExpectedNativeClientT:
        if not isinstance(native_client, expected_native_client_cls):
            raise ValueError(
                f"client_type={client_type!r} requires {expected_native_client_cls.__name__}, got {type(native_client).__name__}"
            )
        return cast(ExpectedNativeClientT, native_client)

    match client_type:
        case "openai_completions":
            native_client = raise_on_invalid_client(
                client_type, native_client, AsyncOpenAI
            )
            return OpenAICompletionsClient(native_client)
        case "openai_chat_completions":
            native_client = raise_on_invalid_client(
                client_type, native_client, AsyncOpenAI
            )
            return OpenAIChatCompletionsClient(native_client)
        case "openai_chat_completions_token":
            native_client = raise_on_invalid_client(
                client_type, native_client, AsyncOpenAI
            )
            return OpenAIChatCompletionsTokenClient(native_client)
        case "anthropic_messages":
            native_client = raise_on_invalid_client(
                client_type, native_client, AsyncAnthropic
            )
            return AnthropicMessagesClient(native_client)
        case _:
            raise ValueError(f"Unsupported client type: {client_type}")


__all__ = [
    "AnthropicMessagesClient",
    "OpenAICompletionsClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "Client",
]
