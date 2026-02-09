from __future__ import annotations

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from typing import cast

from verifiers.clients.client import Client
from verifiers.clients.anthropic import AnthropicMessagesClient
from verifiers.clients.openai import (
    OAIChatCompletionsClient,
    OAIChatCompletionsTokenClient,
    OAICompletionsClient,
)
from verifiers.types import MessageType

CLIENT_REGISTRY = {
    "openai": {
        "completions": OAICompletionsClient,
        "chat_completions": OAIChatCompletionsClient,
        "chat_completions_tokens": OAIChatCompletionsTokenClient,
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
                return OAIChatCompletionsTokenClient(client)
            return OAIChatCompletionsClient(client)
        elif message_type == "completion":
            return OAICompletionsClient(client)
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
            return OAICompletionsClient(cast(AsyncOpenAI, client))
        return OAIChatCompletionsClient(cast(AsyncOpenAI, client))


__all__ = [
    "AnthropicMessagesClient",
    "OAICompletionsClient",
    "OAIChatCompletionsClient",
    "OAIChatCompletionsTokenClient",
    "Client",
]
