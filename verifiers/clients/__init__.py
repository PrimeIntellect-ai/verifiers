from __future__ import annotations

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from typing import cast

from verifiers.clients.client import Client
from verifiers.clients.anthropic_messages_client import AnthropicMessagesClient
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.clients.openai_completions_client import OpenAICompletionsClient
from verifiers.types import MessageType

CLIENT_REGISTRY = {
    "openai": {
        "completions": OpenAICompletionsClient,
        "chat_completions": OpenAIChatCompletionsClient,
        "chat_completions_tokens": OpenAIChatCompletionsTokenClient,
    },
    "anthropic": {"messages": AnthropicMessagesClient},
}


def resolve_client(
    client: Client | AsyncOpenAI | object,
    message_type: MessageType,
    interleaved_rollouts: bool,
) -> Client:
    if isinstance(client, Client):
        return client
    if isinstance(client, AsyncOpenAI):
        if message_type == "chat":
            if interleaved_rollouts:
                return OpenAIChatCompletionsTokenClient(client)
            return OpenAIChatCompletionsClient(client)
        elif message_type == "completion":
            return OpenAICompletionsClient(client)
        else:
            raise ValueError(
                f"Unsupported message type: {message_type} for OpenAI client"
            )
    elif isinstance(client, AsyncAnthropic):
        if message_type == "chat":
            return AnthropicMessagesClient(client)
        elif message_type == "completion":
            raise ValueError(
                "Anthropic does not support raw completion mode. Use message_type='chat' instead."
            )
        else:
            raise ValueError(
                f"Unsupported message type: {message_type} for Anthropic client"
            )
    else:
        # Fall back to OpenAI client for duck-typed clients (e.g., mocks, proxies)
        if message_type == "completion":
            return OpenAICompletionsClient(cast(AsyncOpenAI, client))
        return OpenAIChatCompletionsClient(cast(AsyncOpenAI, client))


__all__ = [
    "AnthropicMessagesClient",
    "OpenAICompletionsClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "Client",
]
